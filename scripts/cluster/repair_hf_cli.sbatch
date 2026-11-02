#!/usr/bin/env bash
# Targeted pure-Python CLI repair; no Torch/CUDA changes or GPU request.
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --job-name=jpa-hf-cli
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/projects/jointposeaction/logs/%x-%j.out
#SBATCH --error=/projects/jointposeaction/logs/%x-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
module load Miniforge3/24.11.3-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$JPA_ENV"
record="$JPA_ROOT/logs/hf-cli-repair-$SLURM_JOB_ID"
mkdir "$record"
git rev-parse HEAD > "$record/git-commit.txt"
python -m pip freeze --all > "$record/pip-before.txt"
python -m pip --isolated install --only-binary=:all: --no-cache-dir --no-deps \
    --index-url https://pypi.org/simple --report "$record/install.json" click==8.1.8
python -m pip check
hf auth login --help
python -m pip freeze --all > "$record/pip-after.txt"
conda env export --prefix "$JPA_ENV" > "$record/environment.yml"
printf 'CLI startup passed; no authentication performed. Evidence: %s\n' "$record"
