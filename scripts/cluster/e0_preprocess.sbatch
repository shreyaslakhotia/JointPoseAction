#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --job-name=e0-preprocess
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/projects/jointposeaction/logs/%x-%j.out
#SBATCH --error=/projects/jointposeaction/logs/%x-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
module load Miniforge3/24.11.3-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$JPA_ENV"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
: "${JPA_E0_MANIFEST:?Set path to reviewed real source manifest}"
python -m e0.preprocess --manifest "$JPA_E0_MANIFEST" \
    --output "$JPA_ROOT/data/processed/e0-$SLURM_JOB_ID"
