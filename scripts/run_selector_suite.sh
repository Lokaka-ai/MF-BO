#!/usr/bin/env bash
# Sequential 22-task runner, NOT a scheduler; use separate allocated workers for parallelism.
# Usage: run_selector_suite.sh MODE DATASET_ROOT [SERVER_ID] [key=value ...]
set -uo pipefail
MODE=${1:?Specify mode}
DATASET_ROOT=${2:?Specify dataset root}
SERVER_ID=${3:-111}
case "$MODE" in halfshrink_ucb|gp6_global_logei|agent_selector|gp7_ancestry_ucb|gp7_no_ancestry_ucb|laplacian_anchor_logei|gp7_long_horizon_ucb|gp7_short_logei_pibo|gp7_long_horizon_ucb_pibo|gp7_short_ucb_pibo) ;; *) echo "Unknown mode: $MODE" >&2; exit 2;; esac
: "${TIME_LIMIT_SECS:?Set TIME_LIMIT_SECS to match the baseline}"
: "${CPUS_PER_TASK:?Set CPUS_PER_TASK to match the baseline}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/runs_$MODE"
SUMMARY="$ROOT/runs_$MODE/suite_$(date +%Y%m%d_%H%M%S).tsv"
printf 'task\texit_code\n' > "$SUMMARY"
failed=0
while IFS= read -r task; do
  [[ -z "$task" || "$task" == \#* ]] && continue
  bash "$ROOT/scripts/run_selector_task.sh" "$MODE" "$task" "$DATASET_ROOT" "$SERVER_ID" "${@:4}"
  code=$?
  printf '%s\t%s\n' "$task" "$code" >> "$SUMMARY"
  [[ "$code" -eq 130 || "$code" -eq 143 ]] && exit "$code"
  [[ "$code" -ne 0 && "$code" -ne 124 ]] && failed=1
  # A provider/config error in the agent arm must not produce 22 repeated failures.
  if [[ "$MODE" == agent_selector && "$code" -ne 0 && "$code" -ne 124 ]]; then
    echo "Agent run failed; stopping suite. Inspect logs before continuing." >&2
    break
  fi
done < "$ROOT/config/mlebench_lite22.txt"
echo "Task exit codes: $SUMMARY (124 means budget timeout, not proof of success)"
exit "$failed"
