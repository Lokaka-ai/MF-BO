#!/bin/bash
# Run MLEvolve on a single competition task.
# Usage: bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]
set -x

EXP_ID=${1:?Usage: bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]}
dataset_dir=${2:?Usage: bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]}
SERVER_ID=${3:-111}
EXTRA_ARGS=("${@:4}")

# ── Proxy (uncomment & fill in if behind a corporate firewall) ──
# export http_proxy=http://YOUR_PROXY:PORT
# export https_proxy=http://YOUR_PROXY:PORT

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Launch the local grading (format-validation) server ──
export DATASET_DIR="${dataset_dir}"
bash "$ROOT/launch_server.sh" "${SERVER_ID}"

BASE_PORT=5005
GRADING_SERVER_PORT=$((BASE_PORT + SERVER_ID))
export GRADING_SERVER_PORT

echo "Waiting for grading server on port ${GRADING_SERVER_PORT} ..."
MAX_WAIT=30
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://127.0.0.1:${GRADING_SERVER_PORT}/health" > /dev/null 2>&1; then
        echo "Grading server ready (port ${GRADING_SERVER_PORT})."
        break
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done
if [ $WAITED -ge $MAX_WAIT ]; then
    echo "Warning: grading server may not be ready yet, proceeding anyway ..."
fi

# ── Experiment settings ──
MEMORY_INDEX=${MEMORY_INDEX:-0}
start_cpu=${START_CPU_ID:-0}
CPUS_PER_TASK=${CPUS_PER_TASK:-21}
TIME_LIMIT_SECS=${TIME_LIMIT_SECS:-43200}  # default 12 hours

export MEMORY_INDEX
format_time() {
  local t=$1
  echo "$((t/3600))hrs $(((t%3600)/60))mins $((t%60))secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)
export STEP_LIMIT=${STEP_LIMIT:-500}

# ── HuggingFace cache (optional, point to a shared directory) ──
# export HF_ENDPOINT=https://huggingface.co
# export HF_DATASETS_CACHE=/path/to/hf_cache
# export HF_MODELS_CACHE=/path/to/hf_cache
# export HUGGINGFACE_HUB_CACHE=/path/to/hf_cache
# export TRANSFORMERS_CACHE=/path/to/hf_cache


# ── Run the main agent loop ──
RUN_INFO=$(mktemp)
trap 'rm -f "$RUN_INFO"' EXIT
CUDA_VISIBLE_DEVICES=$MEMORY_INDEX timeout --foreground --signal=TERM --kill-after=10s "${TIME_LIMIT_SECS}s" python run.py \
  exp_id="${EXP_ID}" \
  dataset_dir="${dataset_dir}" \
  data_dir="${dataset_dir}/${EXP_ID}/prepared/public" \
  desc_file="${dataset_dir}/${EXP_ID}/prepared/public/description.md" \
  exp_name="${EXP_ID}" \
  start_cpu_id="${start_cpu}" \
  cpu_number="${CPUS_PER_TASK}" \
  agent.time_limit="${TIME_LIMIT_SECS}" \
  agent.steps="${STEP_LIMIT}" \
  "${EXTRA_ARGS[@]}" \
  run_result_file="$RUN_INFO"

RUN_EXIT=$?

if [ $RUN_EXIT -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
elif [ $RUN_EXIT -eq 130 ]; then
  echo "Interrupted."
  exit 130
elif [ $RUN_EXIT -ne 0 ]; then
  echo "Run failed with exit code: $RUN_EXIT"
fi

# ── Post-processing: ensemble top solutions ──
if ! EVALUATION_MODE=$(python - "$RUN_INFO" <<'PY'
import json, sys
try:
    info = json.load(open(sys.argv[1]))
    assert type(info['fixed_evaluation']) is bool
    assert info['log_dir'] and info['workspace_dir']
    print('fixed' if info['fixed_evaluation'] else 'legacy')
except (OSError, ValueError, KeyError, AssertionError):
    sys.exit(1)
PY
)
then
  echo "No valid run-info; skipping post-processing."
  [ "$RUN_EXIT" -eq 0 ] && exit 1
  exit "$RUN_EXIT"
fi
if [ "$EVALUATION_MODE" = fixed ]; then
  echo "Exporting best fixed-evaluated node (exact run paths) ..."
  python -m engine.fixed_evaluation finalize --run-info "$RUN_INFO"
  FINAL_EXIT=$?
  if [ "$RUN_EXIT" -eq 0 ] && [ "$FINAL_EXIT" -ne 0 ]; then
    exit "$FINAL_EXIT"
  fi
  exit "$RUN_EXIT"
fi
if [ "$RUN_EXIT" -eq 0 ]; then
  echo "Legacy submission fusion completed by run.py."
  exit 0
fi
echo "Running submission fusion ..."
python utils/submission_fusion_utils.py \
  --task_id "${EXP_ID}" \
  --run_info "$RUN_INFO"

# Preserve the agent exit status, not the optional post-processing exit status.
exit "$RUN_EXIT"
