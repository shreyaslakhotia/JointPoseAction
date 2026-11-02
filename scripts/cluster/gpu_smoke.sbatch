#!/usr/bin/env bash
# Set --gres on the submission command after live resource verification.
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --job-name=jpa-smoke
#SBATCH --time=00:05:00
#SBATCH --output=/projects/jointposeaction/logs/%x-%j.out
#SBATCH --error=/projects/jointposeaction/logs/%x-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
module load Miniforge3/24.11.3-1
# Explicit prefix avoids activating an unrelated named environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$JPA_ENV"
python scripts/cluster/gpu_validate.py
