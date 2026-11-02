#!/usr/bin/env bash
# CPU installation only; never reserve an idle GPU for package installation.
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --job-name=jpa-install
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --output=/projects/jointposeaction/logs/%x-%j.out
#SBATCH --error=/projects/jointposeaction/logs/%x-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
export CONDA_ENVS_PATH="$JPA_ROOT/environments"
module load Miniforge3/24.11.3-1
source "$(conda info --base)/etc/profile.d/conda.sh"

# An existing/partial prefix requires inspection, never an automatic overwrite.
if [[ -e "$JPA_ENV" ]]; then
    printf 'Environment already exists: %s. Inspect before retrying.\n' "$JPA_ENV" >&2
    exit 1
fi
record="$JPA_ROOT/logs/environment-$SLURM_JOB_ID"
mkdir "$record"
git rev-parse HEAD > "$record/git-commit.txt"
git status --porcelain > "$record/git-status.txt"
if [[ -s "$record/git-status.txt" ]]; then
    printf 'Checkout has modifications; review before installing.\n' >&2
    exit 1
fi
cp reproducibility/environments/e0-requirements.txt "$record/direct-requirements.txt"

conda create --yes --prefix "$JPA_ENV" --override-channels --channel conda-forge python=3.12 pip
conda activate "$JPA_ENV"
# Wheel-only installation avoids system compilers/CUDA builds and uses explicit
# indexes rather than inheriting a workstation's extra-index configuration.
python -m pip --isolated install --only-binary=:all: --no-cache-dir \
    --index-url https://pypi.org/simple \
    --report "$record/scientific-install.json" \
    -r reproducibility/environments/e0-requirements.txt
python -m pip --isolated install --only-binary=:all: --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu130 \
    --report "$record/torch-install.json" torch==2.14.1
python -m pip check
hf auth login --help
python scripts/cluster/validate_environment.py
python scripts/cluster/verify_checkout.py > "$record/source-integrity.json"
python -c 'import pathlib,yaml; paths=list(pathlib.Path("reproducibility").rglob("*.yml"))+list(pathlib.Path("reproducibility").rglob("*.yaml")); [yaml.safe_load(p.read_text()) for p in paths]; print("Parsed YAML:",len(paths))'
conda list --explicit > "$record/conda-explicit.txt"
conda env export --prefix "$JPA_ENV" > "$record/environment.yml"
python -m pip freeze --all > "$record/pip-freeze.txt"
printf 'Environment installation and CPU/source checks passed. Versions: %s\n' "$record"
