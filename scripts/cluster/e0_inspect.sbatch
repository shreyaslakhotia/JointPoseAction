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
source scripts/cluster/use_environment.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$JPA_PYTHON" -m unittest discover -s tests -v
"$JPA_PYTHON" -m scripts.data.inspect_e0_sources --download-approved-splits
