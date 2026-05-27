#!/usr/bin/env bash
# Visual Cortex Reconstructor — environment + data bootstrap.
#
# Idempotent. Re-runnable. Uses the existing .venv/ in this directory
# (do NOT recreate — user-provisioned). Skips work that's already done.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

log() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup:warn]\033[0m %s\n' "$*" >&2; }
err() { printf '\033[1;31m[setup:err]\033[0m %s\n' "$*" >&2; }

# --- 1. Venv guard -----------------------------------------------------------
if [[ ! -x ".venv/bin/python" ]]; then
  err ".venv/bin/python not found. The venv must be provisioned before running this script."
  err "Create one with:   python -m venv .venv"
  exit 1
fi
log "using $(./.venv/bin/python --version) at .venv/"

# --- 2. liblsl system dep (Risk R3) -----------------------------------------
# pylsl wraps liblsl. On Arch the package is community/liblsl. On Debian/Ubuntu
# it ships under the same name. If absent, pylsl falls back to a vendored
# library inside the wheel — which is fine, but we warn so the user knows.
if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q liblsl; then
  log "liblsl detected on system loader path"
else
  warn "liblsl not found on system loader path. pylsl will use the wheel-bundled lib."
  warn "On Arch: sudo pacman -S liblsl   (optional but recommended)"
fi

# --- 3. Python deps ---------------------------------------------------------
log "installing pinned dependencies (pip skips already-installed)"
.venv/bin/pip install --quiet --disable-pip-version-check -r requirements.txt

# --- 4. Data --------------------------------------------------------------
mkdir -p data/raw data/processed checkpoints runs results logs

log "running dataset downloader (Spampinato primary, THINGS-EEG2 fallback)"
.venv/bin/python -m data.download_dataset

log "running ImageNet stimuli downloader (40-class subset)"
.venv/bin/python -m data.download_imagenet

# --- 5. Smoke import check --------------------------------------------------
log "verifying core imports"
.venv/bin/python - <<'PY'
import importlib, sys
mods = ["torch", "transformers", "diffusers", "pylsl", "mne",
        "scipy", "numpy", "yaml", "pydantic", "streamlit",
        "umap", "faiss", "openvino"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except ImportError as e:
        missing.append((m, str(e)))
if missing:
    print("FAIL — missing modules:")
    for m, e in missing:
        print(f"  {m}: {e}")
    sys.exit(1)
print("OK — all core modules import.")
PY

log "done. Next: see docs/implementation_plan.md §11 for the resume point."
