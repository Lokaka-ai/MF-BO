#!/usr/bin/env bash
# Normal 12-hour agent campaign. The short verification uses smoke.sbatch.
#SBATCH --job-name=alphaevolve-gp6-12h
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=21
#SBATCH --mem=32G
#SBATCH --time=12:30:00
#SBATCH --array=0-14%15
#SBATCH --output=.cache/alphaevolve/run-%A_%a.log
set -euo pipefail
source tasks/alphaevolve/env.sh
mapfile -t ALPHAEVOLVE_TASKS < config/alphaevolve15.txt
ALPHAEVOLVE_TASK="${ALPHAEVOLVE_TASKS[$SLURM_ARRAY_TASK_ID]}"
ALPHAEVOLVE_OUTPUT="${ALPHAEVOLVE_OUTPUT:-$PWD/runs/alphaevolve_12h_$SLURM_ARRAY_JOB_ID}"
python -u tasks/alphaevolve/run.py --task "$ALPHAEVOLVE_TASK" \
  --cpus "$SLURM_CPUS_PER_TASK" --output "$ALPHAEVOLVE_OUTPUT/$ALPHAEVOLVE_TASK" "$@"
