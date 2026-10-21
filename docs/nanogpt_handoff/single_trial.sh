#!/usr/bin/env bash
# Evaluate a prepared candidate through the official worker. No agent/LLM calls.
set -euo pipefail
: "${NANOGPT_PROJECT:?Set NANOGPT_PROJECT to your MLEvolve_BO checkout}"
: "${NANOGPT_TRIAL:?Set NANOGPT_TRIAL to a fresh prepared trial directory}"
: "${NANOGPT_GPU:?Set NANOGPT_GPU to your allocated GPU UUID or physical GPU index}"
cd "$NANOGPT_PROJECT"
NANOGPT_PROJECT="$(pwd -P)"
NANOGPT_TRIAL="$(realpath "$NANOGPT_TRIAL")"
[[ -f "$NANOGPT_TRIAL/train.py" && -f "$NANOGPT_TRIAL/handoff_preparation.json" ]]
if [[ -e "$NANOGPT_TRIAL/train.log" || -e "$NANOGPT_TRIAL/result.json" ]]; then
    echo 'Use a new prepared trial directory; existing logs/results are not overwritten.' >&2
    exit 1
fi
NANOGPT_GPU_UUID="$(nvidia-smi -i "$NANOGPT_GPU" --query-gpu=uuid --format=csv,noheader)"
[[ "$NANOGPT_GPU_UUID" == GPU-* && "$NANOGPT_GPU_UUID" != *$'\n'* ]]
mkdir -p .cache/gpu-locks
exec 9>".cache/gpu-locks/$NANOGPT_GPU_UUID.lock"
flock -n 9 || { echo 'This checkout already has a workflow using this GPU.' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$NANOGPT_GPU_UUID"
export AUTORESEARCH_CACHE_DIR="${AUTORESEARCH_CACHE_DIR:-$NANOGPT_PROJECT/.cache/autoresearch}"
export AUTORESEARCH_CPUS="${NANOGPT_CPUS:-4}"
export OMP_NUM_THREADS="$AUTORESEARCH_CPUS"
export OPENBLAS_NUM_THREADS="$AUTORESEARCH_CPUS"
export MKL_NUM_THREADS="$AUTORESEARCH_CPUS"
export RAYON_NUM_THREADS="$AUTORESEARCH_CPUS"
export TORCHINDUCTOR_COMPILE_THREADS="$AUTORESEARCH_CPUS"
export PYTHONUNBUFFERED=1
unset EVOLVEFOLD_TRIAL_CONFIG
NANOGPT_COMPILE_DIR="$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/nanogpt-handoff.XXXXXX")"
export TRITON_CACHE_DIR="$NANOGPT_COMPILE_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$NANOGPT_COMPILE_DIR/torchinductor"
export CUDA_CACHE_PATH="$NANOGPT_COMPILE_DIR/cuda"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
.venv/bin/python tasks/autoresearch/manifest.py --verify
printf 'Official trial log: %s/train.log\nCompile cache: %s\n' "$NANOGPT_TRIAL" "$NANOGPT_COMPILE_DIR"
.venv/bin/python tasks/autoresearch/worker.py "$NANOGPT_TRIAL" >"$NANOGPT_TRIAL/train.log" 2>&1
export NANOGPT_TRIAL
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
from tasks.autoresearch.adapter import validate_result
p = Path(os.environ['NANOGPT_TRIAL'])
r = json.loads((p / 'result.json').read_text())
validate_result(r, (p / 'train.py').read_text(), r['total_seconds'])
print(json.dumps({k: r[k] for k in ['valid', 'val_bpb', 'training_seconds', 'num_steps', 'total_tokens', 'eval_tokens', 'peak_vram_mb']}, indent=2))
PY
