#!/usr/bin/env bash
# Bootstrap Granite Docling vLLM on a fresh RunPod (Linux + CUDA) pod.
#
# Usage (inside the repo root, on the pod):
#   bash setup.sh
#
# What it does:
#   1. apt-installs poppler-utils (pdf2image dep).
#   2. pip-installs vllm + docling-core + friends.
#   3. Downloads ibm-granite/granite-docling-258M (untied branch) to ./models/.
#   4. Runs a synthetic end-to-end check (generates a test image, runs the
#      pipeline, asserts known strings appear in the markdown output).
#
# Env knobs:
#   PY_BIN        — python interpreter to use (default: python3)
#   SKIP_APT=1    — skip the apt step (useful if poppler-utils is already there)
#   SKIP_CHECK=1  — skip the synthetic check (download-only)
#   HF_TOKEN      — forwarded to `hf download` if set

set -euo pipefail

PY_BIN="${PY_BIN:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_REPO="ibm-granite/granite-docling-258M"
MODEL_REVISION="untied"
MODEL_DIR="${REPO_ROOT}/models/granite-docling-258M-untied"

log() { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

cd "${REPO_ROOT}"

# --- 1. OS deps -------------------------------------------------------------
if [[ "${SKIP_APT:-0}" != "1" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    log "installing poppler-utils via apt"
    if [[ "$(id -u)" -eq 0 ]]; then
      apt-get update -qq
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends poppler-utils >/dev/null
    else
      sudo apt-get update -qq
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends poppler-utils >/dev/null
    fi
  else
    log "apt-get not found — assuming poppler-utils is already available"
  fi
fi
command -v pdftoppm >/dev/null 2>&1 || die "pdftoppm missing; install poppler-utils manually"

# --- 2. Python deps ---------------------------------------------------------
log "python: $(${PY_BIN} --version)"
${PY_BIN} -m pip install --upgrade pip >/dev/null
log "installing requirements.txt"
${PY_BIN} -m pip install -r requirements.txt

# --- 3. Model download ------------------------------------------------------
if [[ -f "${MODEL_DIR}/config.json" ]]; then
  log "model already present at ${MODEL_DIR} — skipping download"
else
  log "downloading ${MODEL_REPO} (revision: ${MODEL_REVISION}) -> ${MODEL_DIR}"
  ${PY_BIN} -m huggingface_hub download \
      "${MODEL_REPO}" \
      --revision "${MODEL_REVISION}" \
      --local-dir "${MODEL_DIR}" \
      ${HF_TOKEN:+--token "${HF_TOKEN}"}
fi

# --- 4. Quick GPU sanity ----------------------------------------------------
${PY_BIN} - <<'PY'
import torch
print(f"[setup] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[setup] gpu={torch.cuda.get_device_name(0)} count={torch.cuda.device_count()}", flush=True)
PY

# --- 5. Synthetic end-to-end check -----------------------------------------
if [[ "${SKIP_CHECK:-0}" != "1" ]]; then
  log "running synthetic_check.py"
  ${PY_BIN} synthetic_check.py --model-path "${MODEL_DIR}"
else
  log "SKIP_CHECK=1 set — skipping synthetic check"
fi

log "done. Next: python run_ocr.py <your.pdf> -o out/ --model ${MODEL_DIR}"
