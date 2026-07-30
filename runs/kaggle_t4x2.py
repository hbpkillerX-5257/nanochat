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

    !git clone https://github.com/hbpkillerX-5257/nanochat.git
    !python nanochat/runs/kaggle_t4x2.py

Enable Weights & Biases (optional):

    export WANDB_API_KEY=...          # required
    # optional: export WANDB_ENTITY=...  WANDB_PROJECT=nanochat
    python nanochat/runs/kaggle_t4x2.py --wandb
    python nanochat/runs/kaggle_t4x2.py --wandb --wandb-run my-kaggle-d8

Without --wandb, logging stays off (same as --run=dummy).

Disk budget (approx, free tier)
-------------------------------
/kaggle/working is ~20GB and is the only durable output dir.
Large temps go under /kaggle/tmp or /tmp (ephemeral, usually larger).

Typical usage with defaults:
  ClimbMix 6 train + 1 val shards  ~ 0.7 GB
  SFT hub cache (smoltalk+mmlu+gsm8k) ~ 1.0–1.5 GB
  d8 checkpoints (final only)      ~ 1–2 GB
  missing pip wheels only          ~ 0.1–0.5 GB (reuses Kaggle torch)
  TOTAL                             well under /kaggle/working 20GB

Install policy (important on Kaggle):
  - Uses the *system* Python (preinstalled CUDA torch) — no fresh venv
  - Only pip-installs packages whose import currently fails
  - Never reinstalls torch if CUDA torch already works
  - Puts nanochat on PYTHONPATH (no editable install)

Tune STAGE_* / CFG below before running.
"""

from __future__ import annotations

import argparse
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
REPO_URL = "https://github.com/hbpkillerX-5257/nanochat.git"
REPO_DIR_NAME = "nanochat"

# Wandb: set by CLI --wandb / --wandb-run (see parse_args). Defaults = off.
USE_WANDB = False
WANDB_RUN_NAME = MODEL_TAG  # passed as --run=... to base_train / chat_sft

# Force free-tier friendly env
os.environ.setdefault("NANOCHAT_DTYPE", "float16")  # T4 has no bf16
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Avoid huge HF home under /root if possible (set after cache root chosen)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Abort if free disk on the cache volume falls below this (GB)
MIN_FREE_GB = 2.0


# =============================================================================
# CLI / wandb
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="nanochat Kaggle 2xT4 runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging. Requires WANDB_API_KEY in the environment.",
    )
    p.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        help="Wandb run name (and nanochat --run=...). Default: MODEL_TAG. Implies --wandb.",
    )
    return p.parse_args(argv)


def configure_wandb(enable: bool, run_name: str | None) -> str:
    """
    Configure wandb for child training processes.

    Returns the value to pass as --run= to base_train / chat_sft.
    - disabled → "dummy" (nanochat skips wandb)
    - enabled  → run name; reads WANDB_API_KEY from env (export WANDB_API_KEY=...)
    """
    global USE_WANDB, WANDB_RUN_NAME

    if not enable and not run_name:
        USE_WANDB = False
        WANDB_RUN_NAME = "dummy"
        # Keep offline/disabled so accidental wandb.init does nothing useful
        os.environ["WANDB_MODE"] = "disabled"
        print("wandb: OFF  (pass --wandb to enable; key via export WANDB_API_KEY=...)")
        return "dummy"

    USE_WANDB = True
    WANDB_RUN_NAME = run_name or MODEL_TAG
    if WANDB_RUN_NAME == "dummy":
        raise SystemExit("--wandb-run cannot be 'dummy' (that name disables logging in nanochat)")

    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "wandb enabled but WANDB_API_KEY is not set.\n"
            "  export WANDB_API_KEY=...   # from https://wandb.ai/authorize\n"
            "  python runs/kaggle_t4x2.py --wandb"
        )

    # Clear disabled mode so wandb can actually sync.
    # Leave WANDB_MODE=offline alone if the user set it intentionally.
    if os.environ.get("WANDB_MODE") == "disabled":
        del os.environ["WANDB_MODE"]

    # Silence interactive prompts on Kaggle
    os.environ.setdefault("WANDB_SILENT", "true")

    entity = os.environ.get("WANDB_ENTITY", "")
    project = os.environ.get("WANDB_PROJECT", "nanochat")
    print(
        f"wandb: ON  run={WANDB_RUN_NAME!r}  "
        f"entity={entity or '(default)'}  project={project}  "
        f"key=***{api_key[-4:]}"
    )
    return WANDB_RUN_NAME


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


# Package name (pip) -> import name. Only install if import fails.
# `kernels` is optional: FA3 does not run on T4; SDPA fallback is fine.
REQUIRED_IMPORTS: list[tuple[str, str]] = [
    ("torch", "torch"),
    ("filelock", "filelock"),
    ("numpy", "numpy"),
    ("psutil", "psutil"),
    ("pyarrow", "pyarrow"),
    ("rustbpe", "rustbpe"),
    ("tiktoken", "tiktoken"),
    ("wandb", "wandb"),
    ("requests", "requests"),
]
OPTIONAL_IMPORTS: list[tuple[str, str]] = [
    ("kernels", "kernels"),  # FlashAttn3 hub loader; skip quietly on T4
]


def _import_ok(py: str | Path, import_name: str, env: dict | None = None) -> bool:
    r = subprocess.run(
        [str(py), "-c", f"import {import_name}"],
        capture_output=True, text=True, env=env or os.environ.copy(),
    )
    return r.returncode == 0


def _probe_torch(py: str | Path, env: dict | None = None) -> tuple[bool, str]:
    r = subprocess.run(
        [str(py), "-c",
         "import torch; print(torch.__version__); print(torch.cuda.is_available()); "
         "print(torch.cuda.device_count())"],
        capture_output=True, text=True, env=env or os.environ.copy(),
    )
    out = (r.stdout or r.stderr or "").strip()
    lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
    # lines: version, cuda_bool, device_count
    ok = r.returncode == 0 and len(lines) >= 2 and lines[1] == "True"
    return ok, out


def _ensure_pip(py: Path) -> None:
    r = subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True, text=True)
    if r.returncode == 0:
        return
    get_pip = Path("/tmp/get-pip.py")
    print("Bootstrapping pip via get-pip.py …")
    run(["curl", "-fsSL", "https://bootstrap.pypa.io/get-pip.py", "-o", str(get_pip)])
    run([str(py), str(get_pip)])
    get_pip.unlink(missing_ok=True)


def _pip_install_missing(
    py: Path,
    pip_specs: list[str],
    *,
    target: Path | None = None,
    extra_args: list[str] | None = None,
) -> None:
    """Install only the given specs. Never uses --upgrade on the whole env."""
    if not pip_specs:
        print("  nothing to install (all present)")
        return
    cmd = [str(py), "-m", "pip", "install", "--no-cache-dir"]
    # Do not upgrade already-satisfied deps of these packages
    cmd += ["--upgrade-strategy", "only-if-needed"]
    if target is not None:
        # Isolate new wheels under cache; never touch system site-packages
        cmd += ["--target", str(target)]
    else:
        # Prefer user site if not root-writable
        cmd += ["--user"]
    if extra_args:
        cmd += extra_args
    cmd += pip_specs
    print(f"  installing missing: {', '.join(pip_specs)}")
    run(cmd)


def _nuke_broken_venv(cache_root: Path) -> None:
    """Remove half-broken venvs from earlier runner versions (wrapt/sitecustomize hell)."""
    venv = cache_root / "venv"
    if not venv.exists():
        return
    print(f"Removing previous broken/isolated venv at {venv} (we use system Python on Kaggle)")
    shutil.rmtree(venv, ignore_errors=True)


def install_deps(repo: Path, cache_root: Path) -> Path:
    """
    Kaggle-first install: reuse the image Python + preinstalled CUDA torch.

    - No fresh venv (Kaggle venv is broken: no ensurepip / sitecustomize wrapt)
    - No editable `pip install -e .` (build backend flaky) → PYTHONPATH=repo
    - Only pip-install packages whose import currently fails
    - Never reinstall torch if CUDA torch already imports
    - Optional `kernels` (FA3) skipped if install fails — T4 uses SDPA anyway
    """
    require_free(cache_root, min_free_gb=2.0)
    _nuke_broken_venv(cache_root)

    py = Path(sys.executable)
    target = cache_root / "pydeps"
    target.mkdir(parents=True, exist_ok=True)

    # PYTHONPATH: repo first (nanochat package), then our extra wheels
    def build_pythonpath() -> str:
        parts = [str(repo), str(target)]
        prev = os.environ.get("PYTHONPATH", "")
        if prev:
            parts.append(prev)
        return os.pathsep.join(parts)

    os.environ["PYTHONPATH"] = build_pythonpath()
    env = os.environ.copy()

    print(f"Using system python: {py}")
    print(f"Extra packages dir:  {target}")
    print(f"PYTHONPATH={os.environ['PYTHONPATH']}")

    _ensure_pip(py)

    # --- torch: never reinstall if CUDA works ---
    ok, probe = _probe_torch(py, env=env)
    print("torch probe:", " | ".join(probe.splitlines()) if probe else "(empty)")
    if ok:
        print("✓ Reusing preinstalled CUDA torch (will NOT reinstall ~1GB of wheels)")
    else:
        print("CUDA torch not available on system Python — installing torch cu124 once…")
        require_free(cache_root, min_free_gb=5.0)
        try:
            _pip_install_missing(
                py,
                ["torch==2.6.0"],
                target=target,
                extra_args=["--index-url", "https://download.pytorch.org/whl/cu124"],
            )
        except SystemExit:
            print("cu124 index failed; trying default PyPI torch…")
            _pip_install_missing(py, ["torch"], target=target)
        env = os.environ.copy()
        ok, probe = _probe_torch(py, env=env)
        print("torch probe after install:", " | ".join(probe.splitlines()))
        if not ok:
            raise SystemExit(
                "torch+cuda still not importable. Enable GPU T4 x2 accelerator and retry."
            )

    # --- required deps: install only missing imports ---
    missing: list[str] = []
    for pip_name, import_name in REQUIRED_IMPORTS:
        if import_name == "torch":
            continue  # handled above
        if _import_ok(py, import_name, env=env):
            print(f"✓ {import_name} already present")
        else:
            print(f"· {import_name} missing → will install {pip_name}")
            missing.append(pip_name)

    if missing:
        _pip_install_missing(py, missing, target=target)
        # refresh env in case pip wrote anything path-related
        os.environ["PYTHONPATH"] = build_pythonpath()
        env = os.environ.copy()
        # re-check
        still = [p for p, i in REQUIRED_IMPORTS if i != "torch" and not _import_ok(py, i, env=env)]
        if still:
            raise SystemExit(f"Still missing after install: {still}")

    # --- optional kernels (skip if heavy/fails) ---
    for pip_name, import_name in OPTIONAL_IMPORTS:
        if _import_ok(py, import_name, env=env):
            print(f"✓ {import_name} already present (optional)")
        else:
            print(f"· optional {import_name} missing — trying install (ok to fail on T4)…")
            try:
                _pip_install_missing(py, [pip_name], target=target)
            except SystemExit:
                print(f"  skipped {pip_name} (FA3 not needed on T4; SDPA fallback is used)")

    # --- sanity: nanochat via PYTHONPATH, not editable install ---
    run([str(py), "-c",
         "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
         "'gpus', torch.cuda.device_count()); "
         "import nanochat, rustbpe, tiktoken, pyarrow; print('nanochat + deps ok')"],
        env=env)

    (cache_root / "python_path.txt").write_text(str(py.resolve()) + "\n")
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
        f"--run={WANDB_RUN_NAME}",
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
    # Distinct run name so SFT doesn't clobber the pretrain wandb run
    sft_run = WANDB_RUN_NAME if WANDB_RUN_NAME == "dummy" else f"{WANDB_RUN_NAME}-sft"
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
        f"--run={sft_run}",
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

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # --wandb-run alone also turns logging on
    enable_wandb = bool(args.wandb or args.wandb_run)
    configure_wandb(enable_wandb, args.wandb_run)

    t0 = time.time()
    print("nanochat Kaggle 2xT4 runner")
    print(f"NANOCHAT_DTYPE={os.environ.get('NANOCHAT_DTYPE')}")
    print(f"wandb enabled={USE_WANDB}  --run={WANDB_RUN_NAME}")

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
        hint = cache_root / "python_path.txt"
        if hint.exists():
            py = Path(hint.read_text().strip())
        else:
            py = Path(sys.executable)
        print(f"STAGE_INSTALL=False — using {py}")

    # Always put repo + pydeps on path (nanochat is not pip-installed)
    pydeps = cache_root / "pydeps"
    path_parts = [str(repo)]
    if pydeps.is_dir():
        path_parts.append(str(pydeps))
    if os.environ.get("PYTHONPATH"):
        path_parts.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(path_parts)
    print(f"Training python: {py}")
    print(f"PYTHONPATH={os.environ['PYTHONPATH']}")

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
