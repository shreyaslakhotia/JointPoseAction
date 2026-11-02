#!/usr/bin/env bash
# CPU-only, user-approved 183.30 MB metadata plus <=4 MB bounded archive ranges.
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --job-name=e0-inspect
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
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
python -m unittest discover -s tests -v
python -m scripts.data.inspect_e0_sources --download-approved-splits
