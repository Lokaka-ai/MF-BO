#!/usr/bin/env bash
# Submit from the MLEvolve_BO directory; safe alongside the general preparer.
#SBATCH --job-name=mlebench-lite-large
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=.cache/mlebench-lite/large-prepare-slurm-%j.log
set -euo pipefail
source tasks/mlebench_lite/env.sh
python -u tasks/mlebench_lite/prepare.py --prepare-only --wait-for-downloads \
  --workers 1 --tasks new-york-city-taxi-fare-prediction siim-isic-melanoma-classification \
  --status-file .cache/mlebench-lite/large-prepare-status.json
