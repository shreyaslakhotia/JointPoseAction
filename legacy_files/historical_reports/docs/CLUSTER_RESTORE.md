# Restoring the preserved project on a new machine

These commands are a **future restore recipe**. They were not executed against
NTU EEE Cluster 02 during preservation. Use cluster-approved storage, modules,
GPU allocation and internet/download policy; no cluster-specific scheduler,
filesystem, hostname or credentials have been assumed. Start from a fresh clone.

## 1. Clone the custom code and restore pinned upstream repositories

```bash
git clone https://github.com/shreyaslakhotia/JointPoseAction.git
cd JointPoseAction
project_root="$PWD"

git clone https://github.com/facebookresearch/assemblyhands-toolkit.git assemblyhands-toolkit
git -C assemblyhands-toolkit checkout --detach 8d4b7f1bce3f99874514199aae7504f78ebf3517

git clone https://github.com/facebookresearch/sam-3d-objects.git 3D_reconstruction/single_view/sam3d_repo
git -C 3D_reconstruction/single_view/sam3d_repo checkout --detach f91db411c50efee93d8db7aeb323885650f6f722

git clone https://github.com/facebookresearch/sam2.git ../sam2
git -C ../sam2 checkout --detach 2b90b9f5ceec907a1c18123530e92e794ad901a4

git -C 3D_reconstruction/single_view/sam3d_repo apply --check "$project_root/reproducibility/upstreams/sam3d-objects/local-modifications.patch"
git -C 3D_reconstruction/single_view/sam3d_repo apply "$project_root/reproducibility/upstreams/sam3d-objects/local-modifications.patch"
cp reproducibility/upstreams/sam3d-objects/custom_inference.py 3D_reconstruction/single_view/sam3d_repo/custom_inference.py
mkdir -p 3D_reconstruction/single_view/sam3d_repo/checkpoints/hf
cp reproducibility/upstreams/sam3d-objects/checkpoints/hf/*.yaml 3D_reconstruction/single_view/sam3d_repo/checkpoints/hf/
```

Run patch application once on the specified clean revision. A second application
should fail rather than silently change an unrelated version. `upstreams.json`
records all three repository identities. The toolkit checkout supplies the
original calibration examples; it does not supply refined AssemblyHands poses.
The small mistake CSV snapshots are already present in the project clone.

The SAM3D custom file and patches preserve the experiment **including its known
bugs**. Do not treat restored source as successful inference. No whole upstream
repository, submodule/gitlink, checkpoint or `.git` directory is embedded in this
backup commit.

## 2. Recreate environments, with cluster-compatible builds

Environment evidence is in `reproducibility/environments/`:

- `*.observed.yml`: complete Conda export with builds/prefix removed; retains
  exact observed versions, including package names that were installed editable.
- `*.restore.yml`: same version snapshot, omitting SAM2/SAM3D editable package
  entries (install from restored source below) and replacing known pip VCS builds
  with exact commits from `*.direct-origins.json`.
- `fyp-pose.history.yml`: original minimal Conda request (Python 3.10). SAM3D
  `--from-history` failed with `CondaValueError: Requested package
  'zstd==1.5.7=hb8e6e7a_2' is not found in 'explicit_packages'`; no fabricated
  history file is provided.
- SAM3D original upstream `environments/default.yml`, requirements, setup and
  pyproject snapshot are under `reproducibility/upstreams/sam3d-objects/reference/`.

For a version-snapshot restoration attempt on compatible Linux hardware:

```bash
# From the project root; use the cluster's Conda/mamba installation.
# fyp-pose on gpu2 had torch 2.14.1+cu130 at runtime.
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130
conda env create -f reproducibility/environments/fyp-pose.restore.yml
conda activate fyp-pose
pip install --no-deps -e ../sam2
conda deactivate

export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS=https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
conda env create -f reproducibility/environments/sam3d-objects.restore.yml
conda activate sam3d-objects
pip install --no-deps -e ./3D_reconstruction/single_view/sam3d_repo
```

This is an unexecuted version-recreation recipe, not a promise that every
historical wheel remains available or matches the cluster GPU/driver ABI.
Observed exports contain many transitive packages. If solving/building fails,
retain the snapshot for diagnosis and use the **pinned upstream setup** as the
alternative starting specification, documenting changes. The original SAM3D
setup uses staged `pip install -e '.[dev]'`, `'.[p3d]'`, then `'.[inference]'`;
its PyTorch3D/Kaolin/FlashAttention/spconv components need compatible CUDA builds.
Do not install a similarly named SAM package from PyPI in place of the checkout.

SAM3D's original setup also calls `./patching/hydra` from its repository root.
That pinned script requires hydra-core 1.3.2 and replaces `hydra/core/utils.py`
from commit `78f00766b5f37672aa7232ebbf01bdd74246bd60` of `gleize/hydra`.
Follow the original setup for this compatibility patch when recreating that
stack; distribution version alone cannot establish whether it has been applied.

On gpu2 the driver could not expose usable CUDA to either environment at audit
time. No working inference environment was validated. SAM3D upstream documents
≥32 GB VRAM; restoring the four-way wrapper does not resolve its initialization,
tensor-device, PIL-adapter or export defects. Check bounded imports/CUDA status
before attempting research inference in a later phase.

## 3. Reacquire data and weights directly on cluster storage

Use approved dataset/model access and authentication on the cluster, not tokens
copied into this repository. Dataset/model terms and access approvals still
apply. No data/model download is performed by the preservation task.

The exact desired 20 recordings and eight video paths each are listed in
`reproducibility/datasets/assembly101_selection.json`. Additional 67 pose-only
recordings are in `legacy_pose_inventory.json`. Example HF CLI commands for a
**selected manifest recording** (replace the placeholder, repeat only as needed):

```bash
hf download cvml-nus/assembly101 --repo-type dataset --include 'recordings/<recording-id>/C*_rgb.mp4' --local-dir data/raw_videos
hf download cvml-nus/assembly101 'poses@60fps/<recording-id>.json' --repo-type dataset --revision bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967 --local-dir data/poses
```

Raw video revision is unknown: record the resolved revision and checksums for
new downloads. Do not assume the pose revision proves video provenance. The
old `scripts/download_dataset.py` remains unchanged: its main program requests
100 sorted CSV recordings, has no selection-manifest interface, and its
calibration ZIP `/blob/` URL is defective. Its catch-and-continue completion
message is not evidence of successful calibration acquisition. Use its video/
pose logic as historical reference; do not run it blindly to restore only 20.

SAM2: download **only** the large 2.1 checkpoint required by current code from
the URL in `MIGRATION_MANIFEST.md` into `../sam2/checkpoints/`; the upstream
download shell script would fetch all model sizes if run without changes.

SAM3D: follow the restored pinned `doc/setup.md` for approved model access and
download checkpoint files directly from `facebook/sam-3d-objects`. Its HF layout
places files under `checkpoints/`; place weights at the paths named in the
preserved `checkpoints/hf/pipeline.yaml`. The workstation HF cache ref was
`2e73555018d2741ccd486e56c24fac41155a1dc6`, but local checkpoint equivalence
was not established. Restore saved YAMLs after acquiring weights if the model
download has supplied different configuration files, and record all differences.
Do not recreate the two zero-byte placeholder weights as usable checkpoints.

The DINO/MoGe models are additional dependencies; use official sources listed
in the migration manifest and record resolved source/weight identities. Keep
new weights, caches and raw videos outside Git. The source-frame JPEGs and
small original mistake annotations needed for the preserved experiments are
already backed up, so they do not need a workstation transfer.

## 4. Existing preprocessing, without inventing a new extractor

No source implementation was refactored. First review hardcoded `/home/mlda/`
project/SAM2 paths in `scripts/package_reconstruction_data.py`,
`extract_sam3_masks.py`, `run_reconstruction.py` and `generate_masks.py`.
They need an explicit cluster path adjustment in a later phase. Relative paths
also depend on the working directory. The old notebook points to absent HMC
inputs and an outdated pose directory.

Existing regeneration steps, once paths/dependencies are explicitly configured:

1. `scripts/extract_sam3_masks.py` reads the preserved eight contour images and
   writes binary masks. Its eight saved outputs are also included as evidence;
   do not overwrite them when merely verifying a clone. Exact reproduction was
   checked in memory during Phase 0.
2. `scripts/package_reconstruction_data.py` reads the preserved JPEGs, binary
   masks and restored toolkit calibration to recreate RGB/mask/RGBA folders.
   Original `session_01/metadata.json` remains in Git; save/compare it before a
   future run that would overwrite it. The regenerated original formats were
   checked for source-RGB/mask/alpha equality in Phase 0.
3. `experiments/sam2_test/generate_masks.py` expects its working directory to be
   `experiments/sam2_test/` and runs independent box-prompted SAM2 inference.
   The old masks/selected overlays are preserved without needing inference.
4. Hand visualizers are preserved as historical code; calibration reuse,
   annotation confidence/joint ordering and synchronization assumptions are
   not validated. Output videos are excluded.

There is no recovered implementation for extracting these 88 JPEGs from raw
recordings, nor a native SAM3 inference implementation. Saved suffix `0005` is
not a known raw frame index. `future_clip_selections` in the selection JSON is
intentionally empty, with required provenance fields described; a future
extractor may consume it, but no new pipeline is implemented now.

## 5. Verify a restored checkout before new work

Confirm exact upstream SHAs, successful patch application, and SHA256 equality
of the custom/config snapshots using `preservation_checksums.json` and
`upstreams.json`. `small_artifacts.json` verifies original retained image/label
bytes. Inspect `reports/phase0_audit/` and resolve its data/geometry gates before
claiming scientific validity. Git-tracked custom source should be clean; local
vendor modifications after restoration should equal the saved patch plus the
restored custom file/configs. Generated outputs, environments and dependencies
are intentionally ignored rather than committed to the main repository.
