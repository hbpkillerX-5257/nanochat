#!/usr/bin/env python3
"""
nanochat on free Kaggle 2x T4 (16GB each) — disk- and VRAM-safe end-to-end run.

What this does
--------------
1) Picks a cache directory with enough free space (prefers /kaggle/tmp over /kaggle/working)
2) Clones / installs nanochat with a small dependency footprint
3) Downloads only a few ClimbMix shards (~100MB each)
4) Trains a small model (default depth=8) with fp16 + small micro-batches
5) Optional short SFT
6) Copies final artifacts into /kaggle/working so you can download them

Kaggle usage
------------
1. New Notebook → Accelerator: GPU T4 x2
2. Internet ON
3. Paste this whole file into a cell, OR upload it and run:

    !python kaggle_t4x2.py

Or from a clone of the repo:

    !git clone https://github.com/karpathy/nanochat.git
    !python nanochat/runs/kaggle_t4x2.py

Disk budget (approx, free tier)
-------------------------------
/kaggle/working is ~20GB and is the only durable output dir.
Large temps go under /kaggle/tmp or /tmp (ephemeral, usually larger).

Typical usage with defaults:
  ClimbMix 6 train + 1 val shards  ~ 0.7 GB
  SFT hub cache (smoltalk+mmlu+gsm8k) ~ 1.0–1.5 GB
  d8 checkpoints (final only)      ~ 1–2 GB
  venv / torch wheels              ~ 4–8 GB (if install needed)
  TOTAL                             stay under ~12 GB if careful

Tune STAGE_* / CFG below before running.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# =============================================================================
# USER CONFIG — edit these for your session
# =============================================================================

# Pipeline stages (set False to skip / resume mid-session)
STAGE_INSTALL = True
STAGE_DOWNLOAD_DATA = True
STAGE_TRAIN_TOKENIZER = True
STAGE_PRETRAIN = True
STAGE_SFT = True          # set False to only pretrain (saves ~1–2GB + time)
STAGE_CHAT_SMOKE = True  # one-shot CLI prompt after SFT (or base if SFT off)
STAGE_EXPORT = True      # copy final artifacts into /kaggle/working

# Model / training (2x T4 safe defaults)
DEPTH = 8
MAX_SEQ_LEN = 1024
DEVICE_BATCH_SIZE = 4          # drop to 2 or 1 if OOM
TOTAL_BATCH_SIZE = 65536       # must be multiple of device_batch * seq * n_gpus
NUM_ITERATIONS = 1500          # time-box pretrain; raise if you have hours left
# If NUM_ITERATIONS is None, uses TARGET_PARAM_DATA_RATIO instead
TARGET_PARAM_DATA_RATIO = 4.0  # default nanochat is ~12; lower = shorter run

# Tokenizer
TOK_MAX_CHARS = 500_000_000    # 0.5B chars (default repo uses 2B)
NUM_CLIMBMIX_SHARDS = 6        # each ~92MB; +1 val shard always

# SFT (only if STAGE_SFT)
SFT_NUM_ITERATIONS = 400       # short; full SmolTalk epoch is huge
SFT_DEVICE_BATCH_SIZE = 2
SFT_MMLU_EPOCHS = 1
SFT_GSM8K_EPOCHS = 1

# Multi-GPU
NPROC = 2                      # Kaggle T4 x2
MODEL_TAG = "kaggle-d8"

# Repo source
REPO_URL = "https://github.com/karpathy/nanochat.git"
REPO_DIR_NAME = "nanochat"

# Force free-tier friendly env
os.environ.setdefault("NANOCHAT_DTYPE", "float16")  # T4 has no bf16
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Avoid huge HF home under /root if possible (set after cache root chosen)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Abort if free disk on the cache volume falls below this (GB)
MIN_FREE_GB = 2.0


# =============================================================================
# Helpers
# =============================================================================

def gb(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def disk_usage(path: str | Path) -> tuple[float, float, float]:
    """Return (total_gb, used_gb, free_gb) for the filesystem containing path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    u = shutil.disk_usage(path)
    return gb(u.total), gb(u.used), gb(u.free)


def report_disk(label: str, paths: list[str | Path]) -> None:
    print("\n" + "=" * 72)
    print(f"DISK @ {label}")
    print("=" * 72)
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  {p}: (missing)")
            continue
        total, used, free = disk_usage(p)
        # directory size (best-effort, may be slow on huge trees — keep shallow)
        try:
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            size_s = f"{gb(size):.2f} GB"
        except Exception:
            size_s = "?"
        print(f"  {p}")
        print(f"    dir_size≈{size_s}  fs_total={total:.1f}G  fs_free={free:.1f}G")
    print("=" * 72 + "\n", flush=True)


def require_free(path: str | Path, min_free_gb: float = MIN_FREE_GB) -> None:
    _, _, free = disk_usage(path)
    if free < min_free_gb:
        raise SystemExit(
            f"Refusing to continue: only {free:.2f} GB free under {path} "
            f"(need ≥ {min_free_gb} GB). Delete old outputs or lower NUM_CLIMBMIX_SHARDS / skip SFT."
        )


def run(cmd: list[str] | str, cwd: str | Path | None = None, env: dict | None = None) -> None:
    if isinstance(cmd, str):
        print(f"\n$ {cmd}", flush=True)
        shell = True
    else:
        print(f"\n$ {' '.join(cmd)}", flush=True)
        shell = False
    merged = os.environ.copy()
    if env:
        merged.update(env)
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged, shell=shell)
    if r.returncode != 0:
        raise SystemExit(f"Command failed ({r.returncode}): {cmd}")


def which(bin_name: str) -> str | None:
    return shutil.which(bin_name)


def pick_cache_root() -> Path:
    """
    Prefer large ephemeral storage for datasets / checkpoints / HF cache.
    Fall back to /kaggle/working if needed.
    """
    candidates = []
    # Kaggle GPU sessions usually expose a roomy /kaggle/tmp
    for p in ("/kaggle/tmp", "/tmp", "/kaggle/working"):
        if Path(p).exists() or p == "/kaggle/working":
            try:
                Path(p).mkdir(parents=True, exist_ok=True)
                total, _, free = disk_usage(p)
                candidates.append((free, total, Path(p)))
            except Exception:
                pass
    if not candidates:
        return Path("/kaggle/working/nanochat_cache")
    candidates.sort(reverse=True)  # most free first
    free, total, root = candidates[0]
    print(f"Selected cache root: {root}  (free={free:.1f} GB / total={total:.1f} GB)")
    return root / "nanochat_cache"


def pick_work_root() -> Path:
    """Durable notebook output (downloadable)."""
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/nanochat_out")
    return Path.cwd() / "nanochat_out"


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return gb(total)


def prune_old_checkpoints(ckpt_dir: Path, keep_last: int = 1) -> None:
    """Keep only the latest model_*.pt (+ matching meta/optim) to save disk."""
    if not ckpt_dir.exists():
        return
    models = sorted(ckpt_dir.glob("model_*.pt"))
    if len(models) <= keep_last:
        return
    victims = models[:-keep_last]
    for m in victims:
        step = m.stem.split("_", 1)[1]
        for pat in (f"model_{step}.pt", f"meta_{step}.json", f"optim_{step}_rank*.pt"):
            for f in ckpt_dir.glob(pat):
                print(f"  pruning {f}")
                f.unlink(missing_ok=True)


# =============================================================================
# Install
# =============================================================================

def ensure_repo(work_root: Path) -> Path:
    """Return path to nanochat repo (clone if this file is standalone)."""
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    # Running from inside a checkout: .../nanochat/runs/kaggle_t4x2.py
    if (here.parent / "pyproject.toml").exists() and (here.parent / "nanochat").is_dir():
        repo = here.parent
        print(f"Using existing repo at {repo}")
        return repo

    # Notebook / uploaded script: clone into work_root parent
    repo = work_root.parent / REPO_DIR_NAME
    if not (repo / "pyproject.toml").exists():
        print(f"Cloning {REPO_URL} → {repo}")
        run(["git", "clone", "--depth", "1", REPO_URL, str(repo)])
    else:
        print(f"Repo already present at {repo}")
    return repo


def _probe_torch(py: str | Path) -> tuple[bool, str]:
    r = subprocess.run(
        [str(py), "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"],
        capture_output=True, text=True,
    )
    out = (r.stdout or r.stderr or "").strip()
    ok = r.returncode == 0 and "True" in out
    return ok, out


def _bootstrap_pip(py: Path) -> None:
    """Install pip into a venv created with --without-pip (common on Kaggle)."""
    # Already have pip?
    r = subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True, text=True)
    if r.returncode == 0:
        return
    get_pip = Path("/tmp/get-pip.py")
    print("Bootstrapping pip via get-pip.py …")
    run(["curl", "-fsSL", "https://bootstrap.pypa.io/get-pip.py", "-o", str(get_pip)])
    run([str(py), str(get_pip)])
    get_pip.unlink(missing_ok=True)


def _create_venv(venv: Path) -> Path:
    """
    Create a venv robustly. Kaggle's system Python often lacks ensurepip, so
    `python -m venv` fails — fall through several strategies.
    Returns path to the venv python.
    """
    py = venv / "bin" / "python"
    if py.exists():
        print(f"Reusing existing venv at {venv}")
        _bootstrap_pip(py)
        return py

    # 1) uv (best if we can install it)
    uv_bin = which("uv")
    if uv_bin is None:
        uv_candidate = venv.parent / "bin" / "uv"
        if not uv_candidate.exists():
            print("Trying to install uv into cache …")
            try:
                env = os.environ.copy()
                env["UV_INSTALL_DIR"] = str(venv.parent / "bin")
                # official standalone installer
                subprocess.run(
                    "curl -fsSL https://astral.sh/uv/install.sh | sh",
                    shell=True, check=True, env=env,
                )
            except Exception as e:
                print(f"  uv install skipped: {e}")
        # install.sh puts uv in ~/.local/bin or UV_INSTALL_DIR
        for cand in (
            venv.parent / "bin" / "uv",
            Path.home() / ".local" / "bin" / "uv",
            Path.home() / ".cargo" / "bin" / "uv",
        ):
            if cand.exists():
                uv_bin = str(cand)
                break
        if uv_bin is None:
            uv_bin = which("uv")

    if uv_bin:
        print(f"Creating venv with uv ({uv_bin}) at {venv}")
        run([uv_bin, "venv", str(venv), "--python", sys.executable])
        if py.exists():
            return py

    # 2) stdlib venv (may fail on Kaggle without ensurepip)
    print(f"Creating venv at {venv}")
    r = subprocess.run([sys.executable, "-m", "venv", str(venv)], capture_output=True, text=True)
    if r.returncode == 0 and py.exists():
        _bootstrap_pip(py)
        return py
    print(f"  plain venv failed: {(r.stderr or r.stdout or '').strip()[:400]}")

    # 3) venv --without-pip + get-pip.py
    if venv.exists():
        shutil.rmtree(venv, ignore_errors=True)
    print(f"Retrying venv --without-pip at {venv}")
    r = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        capture_output=True, text=True,
    )
    if r.returncode == 0 and py.exists():
        _bootstrap_pip(py)
        return py
    print(f"  venv --without-pip failed: {(r.stderr or r.stdout or '').strip()[:400]}")

    # 4) virtualenv package via system pip (if available)
    if venv.exists():
        shutil.rmtree(venv, ignore_errors=True)
    print("Trying virtualenv module …")
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "virtualenv", "-q"], check=False)
    r = subprocess.run(
        [sys.executable, "-m", "virtualenv", str(venv)],
        capture_output=True, text=True,
    )
    if r.returncode == 0 and py.exists():
        return py
    print(f"  virtualenv failed: {(r.stderr or r.stdout or '').strip()[:400]}")

    raise RuntimeError(
        "Could not create a virtualenv on this image. "
        "Will fall back to system Python + --target site-packages."
    )


def _pip_install(py: Path, args: list[str], cwd: Path | None = None, target: Path | None = None) -> None:
    cmd = [str(py), "-m", "pip", "install"]
    if target is not None:
        cmd += ["--target", str(target), "--upgrade"]
    cmd += args
    run(cmd, cwd=cwd)


def install_deps(repo: Path, cache_root: Path) -> Path:
    """
    Install deps and return the Python executable to use for training.

    Strategy (Kaggle-friendly):
    1) Prefer a venv under the large cache volume
    2) If venv creation fails (no ensurepip), use system Python + packages
       installed into cache_root/pydeps via pip --target (reuses Kaggle's
       preinstalled CUDA torch — best for disk + GPU)
    """
    require_free(cache_root, min_free_gb=3.0)

    light = [
        "filelock>=3.19.0",
        "numpy>=1.26.0",
        "psutil>=7.1.0",
        "pyarrow>=21.0.0",
        "rustbpe>=0.1.0",
        "tiktoken>=0.11.0",
        "wandb>=0.21.3",
        "requests",
        "kernels>=0.11.7",
    ]

    py: Path | None = None
    target: Path | None = None  # set when using --target install mode
    mode = "venv"

    try:
        venv = cache_root / "venv"
        py = _create_venv(venv)
        print(f"Using venv python: {py}")
        run([str(py), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
    except Exception as e:
        print(f"Venv path unavailable ({e}); falling back to system Python + --target")
        mode = "target"
        py = Path(sys.executable)
        target = cache_root / "pydeps"
        target.mkdir(parents=True, exist_ok=True)
        # Ensure pip exists on system python
        r = subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True, text=True)
        if r.returncode != 0:
            _bootstrap_pip(py)
        # Put target packages first on path for this process and children
        os.environ["PYTHONPATH"] = str(target) + os.pathsep + os.environ.get("PYTHONPATH", "")
        print(f"Using system python: {py}")
        print(f"Extra packages target: {target}")

    assert py is not None

    # Torch: reuse if CUDA already works (Kaggle images usually have this)
    ok, probe = _probe_torch(py)
    # When using --target, system torch is still importable from site-packages
    if mode == "target":
        ok, probe = _probe_torch(py)
    print("torch probe:", probe.replace("\n", " | "))

    _pip_install(py, light, cwd=repo, target=target)

    if not ok:
        print("Installing torch (CUDA) — this is the big disk hit…")
        require_free(cache_root, min_free_gb=5.0)
        torch_args = ["torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124"]
        try:
            _pip_install(py, torch_args, target=target)
        except SystemExit:
            print("cu124 install failed; trying default PyPI torch…")
            _pip_install(py, ["torch"], target=target)
    else:
        print("Reusing existing CUDA-capable torch (no reinstall)")

    # Install nanochat editable when in a real venv; otherwise rely on PYTHONPATH=repo
    if mode == "venv":
        _pip_install(py, ["-e", ".", "--no-deps"], cwd=repo)
    else:
        # editable install into --target is flaky; put repo on PYTHONPATH instead
        os.environ["PYTHONPATH"] = (
            str(repo) + os.pathsep + str(target) + os.pathsep + os.environ.get("PYTHONPATH", "")
        )
        # drop a .pth so torchrun child processes find packages even if env is stripped
        pth = target / "nanochat_kaggle.pth"
        pth.write_text(f"{repo}\n{target}\n")
        print(f"Wrote path file {pth}")

    # Sanity
    env = os.environ.copy()
    if mode == "target" and target is not None:
        env["PYTHONPATH"] = str(repo) + os.pathsep + str(target) + os.pathsep + env.get("PYTHONPATH", "")
    run([str(py), "-c",
         "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
         "'gpus', torch.cuda.device_count()); "
         "import nanochat; print('nanochat ok')"],
        env=env)

    # Persist resolver hint for STAGE_INSTALL=False restarts
    meta = cache_root / "python_path.txt"
    meta.write_text(str(py.resolve()) + "\n")
    if mode == "target" and target is not None:
        (cache_root / "pydeps_path.txt").write_text(str(target.resolve()) + "\n")
    return py


# =============================================================================
# Training stages
# =============================================================================

def download_climbmix(py: Path, n_shards: int) -> None:
    require_free(os.environ["NANOCHAT_BASE_DIR"], min_free_gb=2.0)
    # dataset module always also fetches the val shard (last id)
    run([str(py), "-m", "nanochat.dataset", "-n", str(n_shards), "-w", "2"])
    data_dir = Path(os.environ["NANOCHAT_BASE_DIR"]) / "base_data_climbmix"
    print(f"ClimbMix dir size ≈ {dir_size_gb(data_dir):.2f} GB  ({data_dir})")


def train_tokenizer(py: Path) -> None:
    require_free(os.environ["NANOCHAT_BASE_DIR"], min_free_gb=1.0)
    run([str(py), "-m", "scripts.tok_train", "--max-chars", str(TOK_MAX_CHARS)])


def _torchrun_launcher(py: Path, nproc: int) -> list[str]:
    """Prefer venv torchrun, else python -m torch.distributed.run."""
    tr = py.parent / "torchrun"
    if tr.exists():
        return [str(tr), "--standalone", f"--nproc_per_node={nproc}"]
    return [str(py), "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}"]


def pretrain(py: Path, repo: Path) -> None:
    require_free(os.environ["NANOCHAT_BASE_DIR"], min_free_gb=2.5)
    nproc = min(NPROC, _cuda_count(py))
    if nproc < 1:
        raise SystemExit("No CUDA GPUs visible — set Accelerator to GPU T4 x2")

    # total_batch_size must divide evenly: device_batch * seq * nproc
    world_tokens = DEVICE_BATCH_SIZE * MAX_SEQ_LEN * nproc
    if TOTAL_BATCH_SIZE % world_tokens != 0:
        raise SystemExit(
            f"TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} not divisible by "
            f"device_batch*seq*nproc={world_tokens}. Adjust TOTAL_BATCH_SIZE."
        )

    cmd = _torchrun_launcher(py, nproc) + [
        "-m", "scripts.base_train", "--",
        f"--depth={DEPTH}",
        f"--max-seq-len={MAX_SEQ_LEN}",
        f"--device-batch-size={DEVICE_BATCH_SIZE}",
        f"--total-batch-size={TOTAL_BATCH_SIZE}",
        f"--model-tag={MODEL_TAG}",
        "--save-every=-1",            # only final ckpt → saves disk
        "--core-metric-every=-1",     # CORE bundle is large / slow; skip on Kaggle
        "--sample-every=500",
        "--eval-every=250",
        "--eval-tokens=262144",       # smaller val eval
        "--run=dummy",
    ]
    if NUM_ITERATIONS is not None:
        cmd.append(f"--num-iterations={NUM_ITERATIONS}")
    else:
        cmd.append(f"--target-param-data-ratio={TARGET_PARAM_DATA_RATIO}")

    run(cmd, cwd=repo)
    ckpt = Path(os.environ["NANOCHAT_BASE_DIR"]) / "base_checkpoints" / MODEL_TAG
    prune_old_checkpoints(ckpt, keep_last=1)
    print(f"Pretrain checkpoints ≈ {dir_size_gb(ckpt):.2f} GB")


def sft(py: Path, repo: Path) -> None:
    require_free(os.environ["NANOCHAT_BASE_DIR"], min_free_gb=2.0)
    nproc = min(NPROC, _cuda_count(py))
    # SFT will download SmolTalk/MMLU/GSM8K into HF cache (~1–1.5GB)
    cmd = _torchrun_launcher(py, nproc) + [
        "-m", "scripts.chat_sft", "--",
        f"--model-tag={MODEL_TAG}",
        f"--device-batch-size={SFT_DEVICE_BATCH_SIZE}",
        f"--num-iterations={SFT_NUM_ITERATIONS}",
        f"--mmlu-epochs={SFT_MMLU_EPOCHS}",
        f"--gsm8k-epochs={SFT_GSM8K_EPOCHS}",
        "--load-optimizer=0",          # skip loading big optim states from base
        "--chatcore-every=-1",
        "--eval-every=200",
        "--eval-tokens=131072",
        "--run=dummy",
    ]
    run(cmd, cwd=repo)
    # prune base optim shards after SFT if present (large, not needed for chat)
    base_ckpt = Path(os.environ["NANOCHAT_BASE_DIR"]) / "base_checkpoints" / MODEL_TAG
    for f in base_ckpt.glob("optim_*.pt"):
        print(f"  removing base optim state {f} to free disk")
        f.unlink(missing_ok=True)
    sft_ckpt = Path(os.environ["NANOCHAT_BASE_DIR"]) / "chatsft_checkpoints" / MODEL_TAG
    prune_old_checkpoints(sft_ckpt, keep_last=1)
    for f in sft_ckpt.glob("optim_*.pt"):
        print(f"  removing sft optim state {f} to free disk")
        f.unlink(missing_ok=True)


def chat_smoke(py: Path, repo: Path) -> None:
    # chat_cli defaults to SFT; use base if we skipped SFT
    source = "sft" if STAGE_SFT else "base"
    run([
        str(py), "-m", "scripts.chat_cli",
        "-i", source,
        "-g", MODEL_TAG,
        "-p", "What is the capital of France? Answer in one short sentence.",
    ], cwd=repo)


def export_artifacts(cache_root: Path, work_root: Path) -> None:
    work_root.mkdir(parents=True, exist_ok=True)
    base = Path(os.environ["NANOCHAT_BASE_DIR"])
    pairs = [
        (base / "tokenizer", work_root / "tokenizer"),
        (base / "base_checkpoints" / MODEL_TAG, work_root / "base_checkpoints" / MODEL_TAG),
        (base / "chatsft_checkpoints" / MODEL_TAG, work_root / "chatsft_checkpoints" / MODEL_TAG),
        (base / "chatrl_checkpoints" / MODEL_TAG, work_root / "chatrl_checkpoints" / MODEL_TAG),
    ]
    # checkpoint subdir names may vary — copy whatever exists matching model tag
    for src_root in (base / "base_checkpoints", base / "chatsft_checkpoints"):
        if src_root.exists():
            for child in src_root.iterdir():
                if child.is_dir() and MODEL_TAG in child.name:
                    pairs.append((child, work_root / src_root.name / child.name))

    seen = set()
    for src, dst in pairs:
        key = str(dst)
        if key in seen:
            continue
        seen.add(key)
        if not src.exists():
            continue
        print(f"Export {src} → {dst}")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # small README for the user
    readme = work_root / "README_export.txt"
    readme.write_text(
        f"nanochat Kaggle export\n"
        f"model_tag={MODEL_TAG}\n"
        f"depth={DEPTH}\n"
        f"NANOCHAT_BASE_DIR during training={base}\n"
        f"To chat later (with repo + this folder as NANOCHAT_BASE_DIR):\n"
        f"  export NANOCHAT_BASE_DIR={work_root}\n"
        f"  python -m scripts.chat_cli --model-tag {MODEL_TAG}\n"
    )
    print(f"Exported artifacts under {work_root} ({dir_size_gb(work_root):.2f} GB)")


def _cuda_count(py: Path) -> int:
    r = subprocess.run(
        [str(py), "-c", "import torch; print(torch.cuda.device_count())"],
        capture_output=True, text=True,
    )
    try:
        return int((r.stdout or "0").strip().splitlines()[-1])
    except Exception:
        return 0


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    t0 = time.time()
    print("nanochat Kaggle 2xT4 runner")
    print(f"NANOCHAT_DTYPE={os.environ.get('NANOCHAT_DTYPE')}")

    work_root = pick_work_root()
    cache_root = pick_cache_root()
    work_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    # All nanochat artifacts (data, tok, ckpts) live on the large volume
    nano_base = cache_root / "base"
    nano_base.mkdir(parents=True, exist_ok=True)
    os.environ["NANOCHAT_BASE_DIR"] = str(nano_base)

    # HuggingFace hub cache (SFT datasets) also on large volume
    hf_home = cache_root / "hf"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")

    report_disk("start", [work_root, cache_root, "/kaggle/working", "/kaggle/tmp", "/tmp"])

    # Expected data size warning
    shard_gb = NUM_CLIMBMIX_SHARDS * 0.092 + 0.092  # + val
    print(f"Planned ClimbMix download ≈ {shard_gb:.2f} GB for {NUM_CLIMBMIX_SHARDS}+val shards")
    if STAGE_SFT:
        print("SFT will also pull ~1–1.5 GB of Hub datasets (smol-smoltalk, mmlu, gsm8k)")

    repo = ensure_repo(work_root)

    if STAGE_INSTALL:
        py = install_deps(repo, cache_root)
    else:
        # Resolve python from a previous install
        hint = cache_root / "python_path.txt"
        venv_py = cache_root / "venv" / "bin" / "python"
        if hint.exists():
            py = Path(hint.read_text().strip())
        elif venv_py.exists():
            py = venv_py
        else:
            raise SystemExit("STAGE_INSTALL=False but no previous python found — set STAGE_INSTALL=True once")
        pydeps = cache_root / "pydeps"
        if pydeps.is_dir():
            os.environ["PYTHONPATH"] = (
                str(repo) + os.pathsep + str(pydeps) + os.pathsep + os.environ.get("PYTHONPATH", "")
            )

    # Make sure scripts resolve package imports from repo (always)
    os.environ["PYTHONPATH"] = str(repo) + os.pathsep + os.environ.get("PYTHONPATH", "")
    print(f"Training python: {py}")

    report_disk("after install", [cache_root, work_root])

    if STAGE_DOWNLOAD_DATA:
        download_climbmix(py, NUM_CLIMBMIX_SHARDS)
        report_disk("after data", [nano_base])

    if STAGE_TRAIN_TOKENIZER:
        train_tokenizer(py)

    if STAGE_PRETRAIN:
        pretrain(py, repo)
        report_disk("after pretrain", [nano_base, cache_root])

    if STAGE_SFT:
        sft(py, repo)
        report_disk("after sft", [nano_base, cache_root])

    if STAGE_CHAT_SMOKE:
        try:
            chat_smoke(py, repo)
        except SystemExit as e:
            print(f"chat smoke failed (non-fatal): {e}")

    if STAGE_EXPORT:
        export_artifacts(cache_root, work_root)
        report_disk("after export", [work_root, cache_root])

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min.")
    print(f"Durable outputs: {work_root}")
    print(f"Ephemeral cache: {cache_root}")
    print("Download the folder under /kaggle/working/nanochat_out from the notebook UI.")


if __name__ == "__main__":
    main()
