#!/usr/bin/env bash
# Supply partition/account/GPU/time on the sbatch command line; see the handoff.
# This file starts one new workflow and never resumes a historical campaign.
set -euo pipefail
: "${NANOGPT_PROJECT:?Set NANOGPT_PROJECT to your MLEvolve_BO checkout}"
: "${NANOGPT_RUNS:?Set NANOGPT_RUNS to a new output root}"
: "${NANOGPT_PRESET:?Set NANOGPT_PRESET to an experiment preset}"
cd "$NANOGPT_PROJECT"
export AUTORESEARCH_CACHE_DIR="${AUTORESEARCH_CACHE_DIR:-$NANOGPT_PROJECT/.cache/autoresearch}"
export MLEVOLVE_CODEX_REASONING_EFFORT="${MLEVOLVE_CODEX_REASONING_EFFORT:-high}"
export MLEVOLVE_CODEX_TIMEOUT_SECONDS="${MLEVOLVE_CODEX_TIMEOUT_SECONDS:-1800}"

# Resolve the GPU from CUDA's allocation visibility, not a node-wide index.
NANOGPT_ALLOCATED_GPU="$(.venv/bin/python - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit('Request exactly one visible GPU for this workflow.')
value = str(torch.cuda.get_device_properties(0).uuid)
if not value.startswith('GPU-'):
    value = 'GPU-' + value
print(value)
PY
)"
NANOGPT_EXTRA=()
if [[ -n "${NANOGPT_DRAFT_POOL:-}" ]]; then
    NANOGPT_EXTRA+=("shared_draft_pool=$NANOGPT_DRAFT_POOL")
fi
exec bash run_autoresearch.sh \
    --gpu "$NANOGPT_ALLOCATED_GPU" \
    --cpus "${SLURM_CPUS_PER_TASK:-4}" \
    --steps "${NANOGPT_STEPS:-500}" \
    --name "${NANOGPT_NAME:-collaborator_$NANOGPT_PRESET}" \
    --runs-dir "$NANOGPT_RUNS" \
    "experiment_preset=$NANOGPT_PRESET" \
    "agent.time_limit=${NANOGPT_SECONDS:-21600}" \
    "agent.seed=${NANOGPT_WORKFLOW_SEED:-42}" \
    "agent.search.parallel_search_num=${NANOGPT_WORKERS:-2}" \
    "${NANOGPT_EXTRA[@]}"
