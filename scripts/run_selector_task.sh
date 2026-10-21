#!/usr/bin/env bash
# Usage: run_selector_task.sh halfshrink_ucb|gp6_global_logei|agent_selector|gp7_ancestry_ucb|gp7_no_ancestry_ucb|laplacian_anchor_logei|gp7_long_horizon_ucb|gp7_short_logei_pibo|gp7_long_horizon_ucb_pibo|gp7_short_ucb_pibo TASK DATASET_ROOT [SERVER_ID] [key=value ...]
set -euo pipefail
MODE=${1:?Specify selector experiment preset}
TASK=${2:?Specify task id}
DATASET_ROOT=${3:?Specify dataset root}
SERVER_ID=${4:-111}
case "$MODE" in halfshrink_ucb|gp6_global_logei|agent_selector|gp7_ancestry_ucb|gp7_no_ancestry_ucb|laplacian_anchor_logei|gp7_long_horizon_ucb|gp7_short_logei_pibo|gp7_long_horizon_ucb_pibo|gp7_short_ucb_pibo) ;; *) echo "Unknown mode: $MODE" >&2; exit 2;; esac
: "${TIME_LIMIT_SECS:?Set TIME_LIMIT_SECS to the same budget as your baseline campaign}"
: "${CPUS_PER_TASK:?Set CPUS_PER_TASK to the same allocation as your baseline campaign}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/run_single_task.sh" "$TASK" "$DATASET_ROOT" "$SERVER_ID" \
  "experiment_preset=$MODE" "log_dir=$ROOT/runs_$MODE" "workspace_dir=$ROOT/runs_$MODE" "${@:5}"
