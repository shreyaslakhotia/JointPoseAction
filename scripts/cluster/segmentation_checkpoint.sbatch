#!/usr/bin/env bash
#SBATCH --job-name=seg-checkpoint
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/projects/jointposeaction/logs/seg-checkpoint-%j.out
#SBATCH --error=/projects/jointposeaction/logs/seg-checkpoint-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
source scripts/cluster/use_environment.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
# A single native checkpoint in the existing dedicated HF cache; no second
# Transformers checkpoint or extra Xet staging/cache is needed.
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
"$JPA_PYTHON" - <<'PY'
import hashlib, json, os, time
from pathlib import Path
from huggingface_hub import get_token, hf_hub_url, get_hf_file_metadata, hf_hub_download
root = Path('/projects/jointposeaction').resolve()
if root != Path('/projects/_ssd/jointposeaction'):
    raise RuntimeError('Dedicated project SSD required')
used = int(os.getxattr(root, 'ceph.dir.rbytes'))
limit = int(os.getxattr(root, 'ceph.quota.max_bytes'))
size = 3450062241
digest = '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
revision = '3c879f39826c281e95690f02c7821c4de09afae7'
if limit-used < size + 2_000_000_000:
    raise RuntimeError('Checkpoint plus 2 GB headroom does not fit dedicated allocation')
output = root/'experiments/segmentation_pilot'/('checkpoint-'+os.environ['SLURM_JOB_ID'])
output.mkdir(exist_ok=False)
report = {'status':'started','repo':'facebook/sam3','revision':revision,'filename':'sam3.pt',
          'expected_bytes':size,'expected_sha256':digest,'quota_used_before':used,
          'quota_limit_bytes':limit,'job_id':os.environ['SLURM_JOB_ID'],'gpu_used':False}
record = output/'checkpoint.json'
record.write_text(json.dumps(report,indent=2)+'\n')
started = time.monotonic()
try:
    token = get_token()
    meta = get_hf_file_metadata(hf_hub_url('facebook/sam3','sam3.pt',revision=revision),token=token)
    if meta.size != size or meta.etag != digest or meta.commit_hash != revision:
        raise RuntimeError('Pinned official checkpoint metadata mismatch')
    print(json.dumps({'action':'download_verified_native_checkpoint','bytes':size,'remaining_quota_before':limit-used}),flush=True)
    path = Path(hf_hub_download('facebook/sam3','sam3.pt',revision=revision,token=token)).resolve()
    if not path.is_relative_to(root) or path.stat().st_size != size:
        raise RuntimeError('Checkpoint location or size mismatch')
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            sha.update(block)
    if sha.hexdigest() != digest:
        raise RuntimeError('Checkpoint SHA-256 mismatch')
    report.update(status='verified',path=str(path),sha256=sha.hexdigest(),bytes=size)
except Exception as error:
    # SDK exception strings can contain signed URLs; never print or persist them.
    report.update(status='failed',error_type=type(error).__name__)
finally:
    report['elapsed_seconds'] = time.monotonic()-started
    report['quota_used_after'] = int(os.getxattr(root,'ceph.dir.rbytes'))
    record.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if report['status'] != 'verified':
    raise SystemExit(1)
PY
