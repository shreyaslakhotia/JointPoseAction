# gpu2 → NTU EEE Cluster 02 preservation manifest

Snapshot: **1 November 2026**. Repository:
https://github.com/shreyaslakhotia/JointPoseAction, branch `main`.
Pre-preservation HEAD: `ac80eb734752677657a2eee203edf83ec1e8ffdc`.
This backup preserves existing functionality and known failures; no ML feature
was implemented, dependency installed, dataset transferred, or cluster accessed.
Use [CLUSTER_RESTORE.md](CLUSTER_RESTORE.md) for restoration order and commands.

## Included in Git

- All original tracked code/notebook, plus previously untracked
  `scripts/extract_sam3_masks.py`, `scripts/package_reconstruction_data.py`, and
  `3D_reconstruction/single_view/sam3d/run_reconstruction.py`, unchanged.
- Byte-identical experimental SAM 3D `custom_inference.py` under
  `reproducibility/upstreams/sam3d-objects/`, plus a patch for both modified
  upstream source files. Original nested clone stays excluded and untouched.
- All nine active checkpoint-directory YAMLs, exact upstream revisions,
  selected upstream environment/setup/reference files and license. No weights.
- Complete Phase 0 reports and diagnostic JSONs under `reports/phase0_audit/`.
  Those describe the pre-migration layout and should be read as dated evidence.
- Full observed environment exports, restoration variants, direct VCS origins,
  and available Conda history specification under `reproducibility/environments/`.
- Small original experiment evidence: all 88 source JPEGs (7,848,119 bytes), eight
  supplied contour PNGs (15,184,469 bytes), all 88 deprecated masks, eight binary
  masks and their overlays, eight bbox images, four representative deprecated
  overlays, and original `session_01/metadata.json`. Existing source-frame and
  SAM 3 provenance gaps are **not** filled with invented identities.
- All 328 existing small mistake CSVs (182,677 bytes) in their original
  `mistake_repo/annots/` path, with attribution and explicit unknown source
  revision. This is a data-only snapshot, not a vendored repository.

The 540 original small images/masks/CSV files total **28,823,391 bytes**; hashes
are in [small_artifacts.json](../reproducibility/small_artifacts.json).
Custom source/configuration snapshot checksums are in
`reproducibility/preservation_checksums.json`. These establish byte preservation,
not correctness of models, masks, calibration, or annotations.

## Upstream repositories

| Dependency | Restore location relative to project | Exact upstream commit | Local custom state |
|---|---|---|---|
| [AssemblyHands toolkit](https://github.com/facebookresearch/assemblyhands-toolkit) | `assemblyhands-toolkit/` | `8d4b7f1bce3f99874514199aae7504f78ebf3517` | Clean; restore calibration examples from this revision |
| [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) | `3D_reconstruction/single_view/sam3d_repo/` | `f91db411c50efee93d8db7aeb323885650f6f722` | `local-modifications.patch` plus custom Python/YAML snapshots |
| [SAM 2](https://github.com/facebookresearch/sam2) | `../sam2/` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Clean; no local source modifications found |

Machine-readable URLs/locations/commits:
[upstreams.json](../reproducibility/upstreams.json). The SAM 3D patch preserves
debug-print changes in `inference_pipeline.py` and `sparse_structure_vae.py`.
It does not falsely claim to repair the model-parallel experiment. Restoration
of the patch against HEAD and byte equality with current modified files were
checked in a temporary directory before commit.

Installed SAM 3D VCS dependencies are pinned in direct-origin metadata:
gsplat `2323de5905d5e90e035f792fe65bad0fedd413e7`, MoGe
`a8c37341bc0325ca99b9d57981cc3bb2bd3e255b`, PyTorch3D
`75ebeeaea0908c5527e7b1e305fbc7681382db47`, utils3d
`3913c65d81e05e47b9f367250cf8c0f7462a0900`. DINO Torch Hub source was cached
as `facebookresearch_dinov2_main` without recoverable Git revision; its cached
source is not represented as a pinned verified revision.

## Excluded assets — reacquire directly on the cluster

| Excluded asset | Workstation location | Size / count | Official reacquisition source |
|---|---|---|---|
| Raw fixed-camera videos | `data/raw_videos/recordings/` | 160 MP4s, **175,955,481,912 bytes** (~175.96 GB decimal) | [Assembly101 HF dataset](https://huggingface.co/datasets/cvml-nus/assembly101), `recordings/<id>/C*_rgb.mp4`; [official downloader](https://github.com/assembly-101/assembly101-download-scripts) |
| Legacy hand-pose data | `data/poses/poses@60fps/` | 87 JSONs, **5,626,354,608 bytes** | Same HF dataset, `poses@60fps/<id>.json`; cached revision below |
| SAM 2.1 large weights | `../sam2/checkpoints/sam2.1_hiera_large.pt` | **898,083,611 bytes** | Pinned SAM2 `checkpoints/download_ckpts.sh`; [official individual weight](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |
| SAM 3D weights | `3D_reconstruction/single_view/sam3d_repo/checkpoints/` | **13,105,834,810 bytes** including extra encoders/duplicate serialization; exact per-file sizes in audit `CHECKPOINT_INVENTORY.md` | [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects), follow pinned setup/access requirements |
| DINO / MoGe cached weights | `/home/mlda/.cache/torch/hub/checkpoints/`, HF cache | DINO 1,217,607,321 bytes; MoGe 1,256,823,446 bytes | [DINOv2](https://github.com/facebookresearch/dinov2), [Ruicheng/moge-vitl](https://huggingface.co/Ruicheng/moge-vitl); MoGe cached snapshot `ad326bfb61facd6c52b5a825bc1e34d7c97d9672` |
| Conda installs, package/build/download caches, compiled extensions | `/home/mlda/miniconda3`, `~/.cache`, vendor `__pycache__`/egg-info | Not copied or exhaustively sized | Recreate from saved environment specifications and pinned source; build CUDA extensions for cluster runtime |
| Miniconda installer | `Miniconda3-latest-Linux-x86_64.sh` | 197,973,354 bytes | Cluster-provided Conda/modules or official Miniconda distribution |
| Packaged RGB/RGBA/copied masks | `experiments/reconstruction_data/session_01/{rgb,rgba,masks}/` | 31,317,500 bytes total | Regenerate using preserved frames, masks, calibration and existing packager; metadata preserved |
| Most deprecated mask overlays | `experiments/sam2_test/deprecated/mask_overlays/` | 88 originally, 17,192,626 bytes; four representatives retained | Visualization can be recreated from preserved RGB/masks; exact historical encoding is not guaranteed |
| Generated videos/geometry/future outputs/temp files/secrets | Output/checkpoint/cache paths and patterns in `.gitignore` | No project reconstruction output found in Phase 0 | Regenerate as applicable; credentials must be configured separately through approved access |

The raw video/pose total is **181,581,836,520 bytes** (~181.58 GB decimal).
**Do not relay these through gpu2 or a Windows PC.** Download directly to cluster
storage after access/setup. Raw video checksums/revision were not recoverable;
container metadata is in the audit inventory. All 87 pose metadata records point
to HF dataset revision **`bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967`**; preserve
that revision when reproducing the legacy inputs. Cached ETags are recorded but
were not verified by rehashing the large pose files.

The SAM 3D HF cache's `main` ref was `2e73555018d2741ccd486e56c24fac41155a1dc6`.
This is a cached model-source candidate, **not proof that the separate local
checkpoint files are byte-identical to that snapshot**. Preserve the included
YAMLs; record newly downloaded model hashes/revisions on the cluster.

## Recording IDs, annotations and future clip configuration

- [assembly101_selection.json](../reproducibility/datasets/assembly101_selection.json)
  lists all **20 exact downloaded recording IDs**, eight relative video paths
  per recording and corresponding pose path. It also provides an empty
  `future_clip_selections` list and records unknown historical extraction
  fields as null. This is configuration, not a new extraction pipeline.
- [legacy_pose_inventory.json](../reproducibility/datasets/legacy_pose_inventory.json)
  lists **87 exact pose filenames**, sizes and cached revision/ETag metadata.
- [Audit dataset inventory](../reports/phase0_audit/dataset_inventory.json)
  records exact per-camera sizes, fps, frame counts and durations; the readable
  audit table contains all full session identifiers.
- Source is legacy `poses@60fps`; neither cleaned `AssemblyPoses.zip` poses nor
  refined AssemblyHands data were found. Those products have different schemas.

## Known restoration limitations

Existing downloader is preserved unchanged: it uses an incorrect `/blob/` ZIP
URL, catches calibration failures, and defaults to **100** lexicographically
selected CSV identifiers rather than these 20 manifest recordings. It is not a
manifest-driven extractor. See restore instructions before executing it.
Hardcoded `/home/mlda/...` paths, missing extraction provenance, cross-session
calibration reuse, unverified raw synchronization/hand ordering, and broken
SAM 3D input/device/export logic remain documented rather than rewritten.

The full environment exports preserve observed metadata; they are not proof of
a successfully solvable cluster environment or working CUDA runtime. SAM 3D's
history-only Conda export failed with an `explicit_packages` metadata error;
its full export and original pinned upstream specification are included instead.
No cluster environment recreation was attempted.

## Decommissioning boundary

After the backup commit is verified on GitHub, no discovered **custom project
source or vendor modification** should depend solely on this workstation. Large
assets, compiled environments/caches, most generated overlays, and redundant
RGB/RGBA packages remain only locally and will disappear on decommissioning.
Dataset/model reacquisition depends on continued source access; original raw
video revision and unpinned DINO cache source may not be recoverable exactly.
Already-missing extraction identity, SAM 3 prompts/model provenance, native SAM 3
masks, historical OOM logs and unavailable evaluation truth cannot be recovered
from this backup. Items outside the project and the supplied adjacent SAM2 repo
were not searched for undisclosed custom work.
