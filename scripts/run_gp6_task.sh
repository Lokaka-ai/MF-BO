#!/usr/bin/env bash
# Usage: bash scripts/run_gp6_task.sh gp6|gp6_draft TASK DATASET_ROOT [SERVER_ID] [key=value ...]
set -euo pipefail
MODE=${1:?Specify gp6 or gp6_draft}
TASK=${2:?Specify task id}
DATASET_ROOT=${3:?Specify dataset root}
SERVER_ID=${4:-111}
case "$MODE" in gp6|gp6_draft) ;; *) echo "Unknown mode: $MODE" >&2; exit 2;; esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/run_single_task.sh" "$TASK" "$DATASET_ROOT" "$SERVER_ID" \
  "experiment_preset=$MODE" "log_dir=$ROOT/runs_$MODE" "workspace_dir=$ROOT/runs_$MODE" "${@:5}"
