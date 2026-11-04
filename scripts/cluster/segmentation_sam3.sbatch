#!/usr/bin/env bash
#SBATCH --job-name=seg-sam3-2d
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=00:45:00
#SBATCH --output=/projects/jointposeaction/logs/seg-sam3-%j.out
#SBATCH --error=/projects/jointposeaction/logs/seg-sam3-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
PILOT="$JPA_ROOT/experiments/segmentation_pilot"
cd "$PILOT/staging"
PYTHON="$JPA_ROOT/environments/sam3-pilot/bin/python"
"$PYTHON" -m unittest discover -s tests -p 'test_segmentation_pilot_*.py' -v
"$PYTHON" scripts/data/prepare_segmentation_diagnostic.py \
  --survey "$PILOT/survey-268336/survey.json" \
  --geometry "$PILOT/geometry-268824.json" \
  --stability "$PILOT/stability-270038/stability.json" \
  --boxes configs/segmentation_diagnostic_boxes.json \
  --output "$PILOT/inputs-$SLURM_JOB_ID"
"$PYTHON" - <<'PY'
import os
from pathlib import Path
root = Path('/projects/jointposeaction').resolve()
if root != Path('/projects/_ssd/jointposeaction'):
    raise RuntimeError('Dedicated SSD required')
free = int(os.getxattr(root, 'ceph.quota.max_bytes')) - int(os.getxattr(root, 'ceph.dir.rbytes'))
if free < 3221225472 + 2_000_000_000:
    raise RuntimeError('3 GiB output reservation plus 2 GB headroom does not fit')
PY
"$PYTHON" -m segmentation_pilot.sam3_runner \
  --images "$PILOT/inputs-$SLURM_JOB_ID/images.json" \
  --source "$JPA_ROOT/upstreams/sam3" \
  --checkpoint "$JPA_ROOT/cache/huggingface/hub/models--facebook--sam3/blobs/9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e" \
  --output "$PILOT/sam3-$SLURM_JOB_ID" \
  --diagnostic-2d-review --max-images 24 --max-candidates 200 \
  --max-output-bytes 3221225472
