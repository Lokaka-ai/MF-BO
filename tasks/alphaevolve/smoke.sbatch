#!/usr/bin/env bash
# Submit from MLEvolve_BO; one task per CPU allocation, no GPU resource request.
#SBATCH --job-name=alphaevolve-gp6
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-14%15
#SBATCH --output=.cache/alphaevolve/slurm-%A_%a.log
set -euo pipefail
source tasks/alphaevolve/env.sh
mapfile -t ALPHAEVOLVE_TASKS < config/alphaevolve15.txt
ALPHAEVOLVE_TASK="${ALPHAEVOLVE_TASKS[$SLURM_ARRAY_TASK_ID]}"
ALPHAEVOLVE_OUTPUT="${ALPHAEVOLVE_OUTPUT:-$PWD/runs/alphaevolve_smoke_$SLURM_ARRAY_JOB_ID}"
python -u tasks/alphaevolve/run.py --task "$ALPHAEVOLVE_TASK" --smoke \
  --cpus "$SLURM_CPUS_PER_TASK" --output "$ALPHAEVOLVE_OUTPUT/$ALPHAEVOLVE_TASK" "$@"
