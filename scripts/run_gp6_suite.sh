#!/usr/bin/env bash
# Sequentially run the SAME 22 task IDs present in the supplied result bundle.
# Does not acquire cluster resources. Use an allocated machine and existing LLM credentials.
# Usage: bash scripts/run_gp6_suite.sh gp6|gp6_draft DATASET_ROOT [SERVER_ID] [key=value ...]
set -uo pipefail
MODE=${1:?Specify gp6 or gp6_draft}
DATASET_ROOT=${2:?Specify dataset root}
SERVER_ID=${3:-111}
case "$MODE" in gp6|gp6_draft) ;; *) echo "Unknown mode: $MODE" >&2; exit 2;; esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/runs_$MODE"
SUMMARY="$ROOT/runs_$MODE/suite_$(date +%Y%m%d_%H%M%S).tsv"
printf 'task\texit_code\n' > "$SUMMARY"
failed=0
while IFS= read -r task; do
  [[ -z "$task" || "$task" == \#* ]] && continue
  bash "$ROOT/scripts/run_gp6_task.sh" "$MODE" "$task" "$DATASET_ROOT" "$SERVER_ID" "${@:4}"
  code=$?
  printf '%s\t%s\n' "$task" "$code" >> "$SUMMARY"
  [[ "$code" -eq 130 ]] && exit 130
  # 124 denotes the explicit timeout budget, not proof of a successful result.
  [[ "$code" -ne 0 && "$code" -ne 124 ]] && failed=1
done < "$ROOT/config/mlebench_lite22.txt"
echo "Suite exit codes: $SUMMARY"
exit "$failed"
