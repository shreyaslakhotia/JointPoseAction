#!/usr/bin/env bash
# CPU-only environment preparation; submit after frame acquisition finishes.
# No checkpoint download, GPU request, inference, training or reconstruction.
#SBATCH --job-name=seg-sam3-env
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/projects/jointposeaction/logs/seg-sam3-env-%j.out
#SBATCH --error=/projects/jointposeaction/logs/seg-sam3-env-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
source scripts/cluster/use_environment.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="" PIP_NO_CACHE_DIR=1
# All writes remain in project SSD. Bound individual new files so an accidental
# heavyweight wheel cannot consume the model/output reserve (Bash units: KiB).
ulimit -f 262144
pilot_env="$JPA_ROOT/environments/sam3-pilot"
sam3_source="$JPA_ROOT/upstreams/sam3"
sam3_revision=f1647e8d0c1d280991f4d1b7d974c2704537fe66
requirements="${JPA_SAM3_REQUIREMENTS:-$PWD/reproducibility/environments/sam3-pilot-requirements.txt}"
report_dir="$JPA_ROOT/experiments/segmentation_pilot/environment-${SLURM_JOB_ID:?}"
mkdir "$report_dir"
export JPA_SAM3_OVERLAY="$pilot_env" JPA_SAM3_SOURCE="$sam3_source"
export JPA_SAM3_REPORT="$report_dir" JPA_SAM3_REQUIREMENTS_FILE="$requirements"
test -f "$requirements"
cp "$requirements" "$report_dir/requirements.txt"

quota_check() {
    "$JPA_PYTHON" - "$1" <<'PY'
import json, os, sys
from pathlib import Path
root = Path('/projects/jointposeaction').resolve()
if root != Path('/projects/_ssd/jointposeaction'):
    raise RuntimeError('Unexpected dedicated SSD target')
used = int(os.getxattr(root, 'ceph.dir.rbytes'))
limit = int(os.getxattr(root, 'ceph.quota.max_bytes'))
minimum = int(sys.argv[1])
print(json.dumps({'dedicated_used_bytes': used, 'dedicated_limit_bytes': limit,
                  'available_bytes': limit-used, 'required_free_bytes': minimum}))
if limit <= 0 or limit-used < minimum:
    raise RuntimeError('Dedicated quota headroom insufficient; no additional allocation authorized')
PY
}
quota_check 6000000000

# Capture the source environment before any overlay install. No base package is
# uninstalled or updated; every pip invocation below uses the overlay interpreter.
"$JPA_PYTHON" - <<'PY'
import importlib.metadata as md
import json, os, sys
from pathlib import Path
import numpy, torch
expected = Path('/projects/jointposeaction/environments/phase1a').resolve()
if Path(sys.prefix).resolve() != expected or torch.__version__ != '2.14.1+cu130':
    raise RuntimeError('Recognition base interpreter/Torch differs from the reviewed environment')
packages = sorted((d.metadata['Name'], d.version, str(d.locate_file(''))) for d in md.distributions())
record = {'prefix': sys.prefix, 'python': sys.version, 'torch': torch.__version__,
          'torch_path': str(Path(torch.__file__).resolve()), 'numpy': numpy.__version__,
          'packages': packages}
(Path(os.environ['JPA_SAM3_REPORT'])/'base_before.json').write_text(json.dumps(record, indent=2)+'\n')
PY

if [[ ! -x "$pilot_env/bin/python" ]]; then
    if [[ -e "$pilot_env" ]]; then
        printf 'Refusing to replace a partial/unrecognized overlay: %s\n' "$pilot_env" >&2
        exit 1
    fi
    "$JPA_PYTHON" -m venv --system-site-packages "$pilot_env"
fi
"$pilot_env/bin/python" - <<'PY'
import os, sys
from pathlib import Path
expected = Path(os.environ['JPA_SAM3_OVERLAY']).resolve()
base = Path('/projects/jointposeaction/environments/phase1a').resolve()
if Path(sys.prefix).resolve() != expected or Path(sys.base_prefix).resolve() != base:
    raise RuntimeError('Overlay prefix/base mismatch; refusing package installation')
PY

# A shallow sparse checkout contains all model code and packaged BPE vocabulary,
# while excluding large demonstration images/videos. No upstream file is patched.
if [[ ! -e "$sam3_source" ]]; then
    git init "$sam3_source"
    git -C "$sam3_source" remote add origin https://github.com/facebookresearch/sam3.git
    git -C "$sam3_source" sparse-checkout init --no-cone
    git -C "$sam3_source" sparse-checkout set --no-cone /sam3/ /pyproject.toml /README.md /LICENSE
    git -C "$sam3_source" fetch --depth=1 --filter=blob:none origin "$sam3_revision"
    git -C "$sam3_source" checkout --detach "$sam3_revision"
fi
test "$(git -C "$sam3_source" remote get-url origin)" = "https://github.com/facebookresearch/sam3.git"
test "$(git -C "$sam3_source" rev-parse HEAD)" = "$sam3_revision"
git -C "$sam3_source" diff --no-ext-diff --exit-code HEAD --
quota_check 4700000000

# --no-deps is intentional: the inherited CUDA Torch is read-only and must not
# be replaced by resolver-selected builds. Iopath is a small pure-Python sdist.
# Bootstrap its small build tools in the overlay first; Python 3.12 venvs do
# not guarantee either setuptools or wheel is present at creation.
"$pilot_env/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
    --no-deps setuptools==80.9.0 wheel==0.45.1 \
    --report "$report_dir/pip-build-tools.json"
"$pilot_env/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
    --no-deps --prefer-binary --no-build-isolation -r "$requirements" \
    --report "$report_dir/pip-install.json"
quota_check 4700000000
"$pilot_env/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
    --no-deps --no-build-isolation -e "$sam3_source" \
    --report "$report_dir/sam3-install.json"
git -C "$sam3_source" diff --no-ext-diff --exit-code HEAD --
quota_check 4700000000

# Record all inherited dependencies, not just the overlay. A conflict is retained
# for diagnosis and blocks readiness; do not force-uninstall base packages.
pip_check_status=0
"$pilot_env/bin/python" -m pip check > "$report_dir/pip-check.txt" 2>&1 || pip_check_status=$?
export JPA_SAM3_PIP_CHECK_STATUS="$pip_check_status"
"$pilot_env/bin/python" -m pip freeze --all > "$report_dir/observed-freeze.txt"
"$pilot_env/bin/python" - <<'PY'
import hashlib, importlib.metadata as md, json, os, sys, traceback
from pathlib import Path
report = Path(os.environ['JPA_SAM3_REPORT'])
overlay = Path(os.environ['JPA_SAM3_OVERLAY']).resolve()
source = Path(os.environ['JPA_SAM3_SOURCE']).resolve()
def tree_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
record = {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'prefix':sys.prefix, 'base_prefix':sys.base_prefix,
          'python':sys.version, 'source_revision':'f1647e8d0c1d280991f4d1b7d974c2704537fe66',
          'source_path':str(source), 'source_sparse_checkout':True,
          'requirements_sha256':hashlib.sha256((report/'requirements.txt').read_bytes()).hexdigest(),
          'pip_check_exit_code':int(os.environ['JPA_SAM3_PIP_CHECK_STATUS']),
          'checkpoint_downloaded':False, 'gpu_requested':False,
          'model_inference_performed':False, 'training_performed':False, 'reconstruction_performed':False}
try:
    import numpy as np
    import torch, torchvision, cv2, av
    from torchvision.ops import nms
    from sam3.model.sam3_image_processor import Sam3Processor
    if Path(sys.prefix).resolve() != overlay:
        raise RuntimeError('Overlay interpreter mismatch')
    before = json.loads((report/'base_before.json').read_text())
    if torch.__version__ != before['torch'] or str(Path(torch.__file__).resolve()) != before['torch_path']:
        raise RuntimeError('Torch was duplicated or changed; expected the read-only recognition base')
    if np.__version__ != '1.26.4' or torchvision.__version__ != '0.29.1+cu130':
        raise RuntimeError('NumPy or matched CUDA torchvision pin mismatch')
    array = np.array([1, 2, 3], dtype=np.float32)
    tensor = torch.from_numpy(array)
    tensor[1] = 7
    np.testing.assert_array_equal(array, [1, 7, 3])
    np.testing.assert_array_equal(tensor.numpy(), array)
    kept = nms(torch.tensor([[0.,0.,2.,2.],[0.,0.,2.,2.]]), torch.tensor([.9,.8]), .5)
    if kept.tolist() != [0] or Sam3Processor.__name__ != 'Sam3Processor':
        raise RuntimeError('CPU torchvision NMS / SAM3 import check failed')
    sizes = {'overlay':tree_bytes(overlay), 'source':tree_bytes(source)}
    record.update({'numpy':np.__version__, 'torch':torch.__version__, 'torch_path':str(Path(torch.__file__).resolve()),
                   'torchvision':torchvision.__version__, 'opencv':cv2.__version__, 'av':av.__version__,
                   'cpu_numpy_bridge':'passed', 'cpu_torchvision_nms':'passed', 'sam3_processor_import':'passed',
                   'storage_bytes':sizes,
                   'packages':sorted((d.metadata['Name'],d.version,str(d.locate_file(''))) for d in md.distributions())})
    if sum(sizes.values()) > 1_000_000_000:
        raise RuntimeError('SAM3 source+overlay exceeded the 1 GB preparation budget')
    if record['pip_check_exit_code']:
        raise RuntimeError('Dependency conflicts recorded in pip-check.txt; review before inference')
    record['status'] = 'cpu_imports_passed_no_checkpoint_or_gpu_inference'
except Exception as error:
    record.update({'status':'failed', 'error_type':type(error).__name__, 'error':str(error)})
    (report/'environment_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    traceback.print_exc()
    raise
(report/'environment_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ('status','numpy','torch','torchvision','storage_bytes')}))
PY

# Prove base package inventory did not change, using the base interpreter itself.
"$JPA_PYTHON" - <<'PY'
import importlib.metadata as md
import json, os, sys
from pathlib import Path
import numpy, torch
report = Path(os.environ['JPA_SAM3_REPORT'])
before = json.loads((report/'base_before.json').read_text())
after = {'prefix':sys.prefix, 'python':sys.version, 'torch':torch.__version__,
         'torch_path':str(Path(torch.__file__).resolve()), 'numpy':numpy.__version__,
         'packages':sorted((d.metadata['Name'],d.version,str(d.locate_file(''))) for d in md.distributions())}
(report/'base_after.json').write_text(json.dumps(after,indent=2)+'\n')
if json.loads(json.dumps(after)) != before:
    raise RuntimeError('Recognition base inventory changed; investigate before any further work')
print(json.dumps({'recognition_base_inventory_unchanged':True, 'report':str(report/'environment_provenance.json')}))
PY
quota_check 4700000000
