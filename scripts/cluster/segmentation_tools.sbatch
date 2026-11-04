#!/usr/bin/env bash
#SBATCH --job-name=seg-tools
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/projects/jointposeaction/logs/seg-tools-%j.out
#SBATCH --error=/projects/jointposeaction/logs/seg-tools-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
source scripts/cluster/use_environment.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
pilot_env="$JPA_ROOT/environments/sam3-pilot"
"$JPA_PYTHON" - <<'PY'
import os
from pathlib import Path
p = Path('/projects/jointposeaction').resolve()
used = int(os.getxattr(p, 'ceph.dir.rbytes'))
limit = int(os.getxattr(p, 'ceph.quota.max_bytes'))
if limit - used < 6_000_000_000:
    raise RuntimeError('Insufficient dedicated quota headroom for pilot tools and later model')
print({'dedicated_used_bytes': used, 'dedicated_limit_bytes': limit})
PY
if [[ ! -x "$pilot_env/bin/python" ]]; then
    "$JPA_PYTHON" -m venv --system-site-packages "$pilot_env"
fi
# This isolated overlay shares read-only base packages and does not alter them.
# Only the bounded frame-decoding dependency is installed at this stage.
"$pilot_env/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
    --only-binary=:all: --no-deps av==16.0.1
"$pilot_env/bin/python" - <<'PY'
import av, json, sys
from pathlib import Path
expected = Path('/projects/jointposeaction/environments/sam3-pilot').resolve()
if Path(sys.prefix).resolve() != expected:
    raise RuntimeError('Isolated pilot interpreter mismatch')
print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'av':av.__version__}))
PY
