# Phase 0 — Independent technical audit

Audit date: 1 November 2026, Asia/Singapore. Project: **Investigating Joint 3D Hand–Object Poses for Procedural Process Understanding**.

Scope: local repository, adjacent `/home/mlda/shreyas_projects/sam2`, representative local dataset samples, installed-package metadata, and official upstream documentation. No source changes, dependency changes, downloads, training, or model inference were performed. Only this reporting directory was created. No complete previous handover was supplied; comparisons below address claims embedded in the audit brief, code comments, and README, rather than attributing unseen statements to the previous agent.

Evidence labels: **V** = directly verified by inspection or a lightweight diagnostic; **I** = inference supported by stated evidence; **U** = unresolved. **Implemented** means code exists; **executed** requires a run record or compatible artifact; **validated** requires a meaningful correctness check; **reproducible** requires recoverable inputs, configuration, and procedure. Artifact existence alone does not prove which model produced it.

## 1. Executive summary

The project has useful data and exploratory preprocessing, but no demonstrated joint metric 3D hand–object pipeline. Twenty recordings have all eight fixed RGB views; 87 legacy pose JSONs and 328 mistake CSVs are present. Eight reconstruction views have RGB, binary masks, RGBA, and calibration metadata. Mask extraction can be reproduced exactly in memory from the supplied contour images. These are verified assets, not evidence of correct metric reconstruction or temporally aligned interaction features. [E1–E7]

The primary scientific risks precede SAM 3D GPU memory: unknown source recording/time for extracted frames, unverified cross-view/video-to-pose synchronization, a calibration file reused outside its recording, unverified hand joint ordering, and a multipart articulated target. Camera calibrations vary across the locally supplied files. The original raw-video release is explicitly distinguished from synchronized rectified AssemblyHands videos by the toolkit. [E3–E6, E10]

The custom SAM 3D path is structurally broken: CPU and Gaussian-removal patches do not match the actual YAML; it accesses a nonexistent pipeline attribute; module wrappers leave device arguments, sparse tensors, decoders, and nonregistered mesh state unresolved; PIL inputs violate the adapter; mesh serialization is incompatible. More VRAM alone will not make this custom entry point correct. The historical OOM cannot be localized to an exact failing allocation because no traceback/run log was found. The present environment additionally reports no usable CUDA devices. [E8–E9, E12]

The recommended next phase starts with one provenance-complete calibrated clip, independent landmark/timing checks, and a rigid component. A conventional visual hull is a credible CPU-friendly metric baseline after those gates. Hand-only versus hand-plus-object event features can then be evaluated without waiting for full 6D tracking. FoundationPose remains a candidate only after resolving its observed-depth dependency. [E10–E13]

## 2. Verified current project status

| Stage | Implemented | Executed evidence | Validated / reproducible status |
|---|---|---|---|
| Subset download | `scripts/download_dataset.py` | 160 video files, 87 pose JSONs | Presence/container metadata verified; remote download not rerun; provenance revisions/checksums absent |
| Hand overlays | Three `scripts/*render*`/visualization scripts plus notebook | Notebook contains historical Windows execution output | No local project overlay MP4 found; no independent 2D error measurements; notebook is stale |
| SAM 2.1 | Image predictor with boxes | 88 deprecated masks and 88 overlays compatible with script | Model invocation history unrecorded; visible contamination; temporal consistency unimplemented |
| Claimed SAM 3 | No local inference implementation found | Eight externally supplied contour images | Model/checkpoint/prompt/run provenance unknown |
| Contour extraction | `scripts/extract_sam3_masks.py` | Eight masks and overlays | Exact pixel reproduction verified for all eight; object accuracy unmeasured |
| Reconstruction packaging | `scripts/package_reconstruction_data.py` | Eight RGB/mask/RGBA triples and metadata | RGB pixels equal named source JPEGs; no source recording/time; no temporal or metric validation |
| SAM 3D | Vendored upstream plus custom script and launcher | Checkpoints; debug edits; empty project output directory | No successful project inference evidence; custom path has deterministic defects |
| DUSt3R / other multi-view | `3D_reconstruction/multi_view/` empty | None found | Planned only in supplied brief |
| Tracking, fusion, procedural model | No project implementation found | None found | Not demonstrated |

See [COMPONENT_STATUS.md](COMPONENT_STATUS.md) for dependencies and detailed failure modes. Findings of absence apply to inspected project/adjacent-repository locations, not every possible external machine or account.

## 3. Repository architecture

### Git and layout

**V:** root `/home/mlda/shreyas_projects/JointPoseAction`; branch `main`; HEAD `ac80eb734752677657a2eee203edf83ec1e8ffdc`; remote `https://github.com/shreyaslakhotia/JointPoseAction`. Nine tracked files; no modified tracked root files, no index conflicts, no `.gitmodules` or registered submodules. Recent commits: `ac80eb7` SAM-2 pipeline; `0719e4b` download limit; `0e6f349` all-view visualizer; `3856b60` restructuring; `c1433db` baseline setup; `e2549f0` initial commit. No incomplete merge evidence was found. [E1]

Pre-audit untracked groups: `3D_reconstruction/`, the 197,973,354-byte Miniconda installer, `experiments/reconstruction_data/`, SAM bbox/deprecated/SAM3 artifact folders, `scripts/extract_sam3_masks.py`, `scripts/package_reconstruction_data.py`. Ignored groups: `data/`, `assemblyhands-toolkit/`, `mistake_repo/`, extracted frames. This report adds only `reports/phase0_audit/`. Root `git status` does not expose modifications inside ignored vendor clones. [E1]

Nested/adjacent repositories:

| Repository | HEAD | State / remote |
|---|---|---|
| `assemblyhands-toolkit` | `8d4b7f1bce3f99874514199aae7504f78ebf3517` | Clean, `main`; facebookresearch/assemblyhands-toolkit |
| `3D_reconstruction/single_view/sam3d_repo` | `f91db411c50efee93d8db7aeb323885650f6f722` | `main`; facebookresearch/sam-3d-objects; two modified tracked files and untracked custom entry point |
| `/home/mlda/shreyas_projects/sam2` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Clean, `main`; facebookresearch/sam2 |
| `mistake_repo` | No independent Git metadata | Data-only directory; `git -C mistake_repo` resolves the parent repository, not an annotation-repository revision |

The historical directories largely exist. `src/`, `download_subset.py`, `data/annotations/`, `data/processed/` in the README do not match the actual project. The real inputs are exocentric, while the README emphasizes four HMC views. There is no root environment lockfile, dependency manifest, or automated project test suite. Upstream toolkit evaluation code is present but is not a project evaluation result. [E1–E3, E10]

### Data flow and code debt

`download_dataset.py` selects sorted mistake CSV names, downloads fixed-camera MP4s and legacy poses. `visualize_single.py`, `render_batch.py`, `render_all_views.py` duplicate calibration/projection and skeleton drawing. `generate_masks.py` consumes already extracted JPEGs; no recoverable frame-extraction implementation/manifests were found. Contour-image extraction feeds reconstruction packaging, which chooses the fifth sorted RGB independently of the mask candidate. The launcher ranks views by foreground area and invokes upstream `demo.py`, rather than `custom_inference.py`. [E2–E8]

Major reproducibility debt:

- Hardcoded workstation paths, session/camera choices, and working-directory-dependent relative paths throughout the custom scripts. SAM 2 requires running from its experiment directory. The notebook refers to a Windows-era path and a pose location that no longer exists.
- No native SAM 3 predictions, confidence scores, prompts, checkpoint version, or run metadata. The filename `session_01` does not encode a recording.
- `download_calibrations_remotely()` uses a Hugging Face `/blob/` HTML URL as a ZIP stream and catches all failures, then proceeds to videos and prints completion. This is a static URL/API mismatch, not a newly executed network failure. ZIP members are selected broadly by name; calibration artifacts under `data/calibrations` are absent. [E2:8–35]
- `render_all_views.py:142–149` uses up to 34 CPU workers, not GPU rendering, and prints “all 400” regardless of actual queue size. It skips outputs by existence alone; incomplete files would be treated as complete. Writer/open failures and decoded-frame totals are not meaningfully checked. [E3]
- Packaging silently substitutes zero focal lengths and identity extrinsics for missing calibration and does not assert mask dimensions/binarity or check `imwrite` return values. Current packaged matrices are populated, but these defaults are unsafe. [E7:104–149]
- `extract_sam3_masks.py:119–129` reduces identity to camera only: multiple times for one camera overwrite the same mask. NumPy uint8 `g + 40` can wrap in the color test; the supplied outputs still reproduce exactly. This is a latent selection defect, not evidence of a specific bad pixel. [E6]

## 4. Dataset and annotation findings

**V:** twenty session folders, each with `C10095`, `C10115`, `C10118`, `C10119`, `C10379`, `C10390`, `C10395`, `C10404` RGB MP4s. All 160 containers report 1920×1080, nominal and average 60/1 fps, and start time zero. No local HMC or sensor-depth stream was found. Videos total 175,955,481,912 bytes; legacy pose JSONs total 5,626,354,608 bytes. All twenty downloaded recordings have matching pose and mistake filenames. [E4]

All video headers probed successfully. Eighteen first/middle/last frames from six representative videos decoded successfully; this does not certify the remaining frames or exclude dropped acquisition frames. See [DATASET_AUDIT.md](DATASET_AUDIT.md) for all session IDs and frame/duration ranges, and `dataset_inventory.json` for exact per-camera values. [E4]

**V:** the hand inputs are `data/poses/poses@60fps/*.json`, lists containing `frame_index`, absolute-looking `timestamp`, `landmarks` with hand keys `0`/`1`, and hand-level `tracking_confidence`. Each sampled hand has 21 xyz points. The code never loads refined `assemblyhands_*_joint_3d_v1-1.json` annotations. Cleaned `AssemblyPoses.zip` contents were not found. The official downloader explicitly recommends its cleaned poses over legacy `poses@60fps`. These three annotation products must not be conflated. [E2–E4, S1]

Three complete pose files were inspected: 9011-b06b (14,242 rows), 9012-c07c (17,275), 9013-c03b (17,693). All have consecutive zero-based indices; timestamp increments are approximately 0.016/0.017 seconds. Both hand keys exist on every row. Uniform samples of 201/201/202 rows contain respectively 3/1/3 all-zero joints and 3/1/3 zero-confidence hands, with no nonfinite sampled coordinates. No per-joint validity is supplied in this schema; the visualizers ignore confidence and possible zero-joint sentinels. Existence of coordinates does not establish validity. [E4]

**I/U:** numeric scale and calibration translations strongly suggest millimetres in a shared mocap world. Exact physical-unit provenance, world axis/gravity convention, hand-ID semantics, and legacy joint-name ordering were not established from the available metadata. The code calls `0` left and `1` right and draws a wrist-rooted consecutive-finger skeleton; neither assumption is validated by a local schema. AssemblyHands uses a different explicitly documented order (right 0–20, left 21–41; wrists 20/41), so its mapping cannot be transferred to legacy JSON merely because both have 21 points. Resolve ordering before fingertip features. [E3–E4, E10:63–85]

An external primary-code cross-check strengthens the ordering concern: HandFormer's Assembly101 cleaned-pose reader names hand keys `0`/`1` left/right and identifies **joint 5 as wrist**, unlike the project's assumed wrist 0. This corroborates the hand labels for that cleaned product and supplies a concrete counterexample to assuming a universal 21-joint order. It does not by itself certify every legacy joint name or that the legacy and cleaned products share a permutation. [S11]

The twenty mistake files contain **224 intervals: 169 correct, 39 mistake, 16 correction**. They contain verbs, two semantic part names, and remarks, but no image masks, 3D object correspondences, 6D poses, or per-frame contact truth. No nonpositive-duration interval was found in these twenty CSVs. Of the local sessions, only 9012-c07c and 9014-a23 appear in the toolkit's AssemblyHands split table (validation and training respectively). The other eighteen have no membership established by that table; this says nothing definitive about Assembly101's action splits. [E4, E10]

## 5. Calibration and synchronization findings

### Camera convention

**V:** twenty Nimble calibration files contain eight exocentric `OpenCV` cameras plus four ego cameras with `OVFishEye62`. Fixed-camera image dimensions match the 1920×1080 video headers. Project code constructs `K=[[fx,0,cx],[0,fy,cy],[0,0,1]]`, distortion `[k1,k2,p1,p2,k3]`, and uses the nested 4×4 matrix without transpose as `Xc=R Xw+t`. Other fixed-camera distortion coefficients in the calibration schema are zero. `camera_center_world=-R.T@t` is correct for that convention. This is ordinary row-indexed matrix storage with column-vector mathematics; OpenCV x-right/y-down/z-forward is the intended camera convention. [E3, E5, E7]

**V:** for all eight cameras in the reused c07c calibration, determinant differs from 1 only at floating precision and `R.T R` error is at most 1.12e-15. Sampled points have positive depth in every view; in-image fractions range approximately 96.75–100%. These tests support algebraic plausibility of direct world-to-camera use, not physical reprojection correctness. No independent 2D landmark ground truth or pixel-error benchmark was found. [E5; `pose_diagnostics.json`]

**V:** each fixed camera has eleven distinct calibration records across the twenty local calibration files. Maximum elementwise 4×4 differences against c07c range 707–975 in translation-scale units; focal changes reach 4.14 pixels. Only the c07c file matches its complete fixed-camera record exactly. “Universal calibration” is therefore unsupported. Only two calibration filenames exactly match the twenty video-session identifiers. Whether another same-day record is valid requires a documented rig/world-frame mapping and reprojection checks, not filename proximity. [E3:13–14, E5]

For ego cameras, the fixed-camera five-coefficient OpenCV path is inappropriate for `OVFishEye62`, and moving-camera extrinsics would need the appropriate time-dependent rig transform. No current HMC pipeline is demonstrated. [E5, E10]

### Timing

**V:** frame counts differ across all twenty eight-view sets. Maximum spread is 17 frames (9012-a17, about 283 ms at 60 fps). b06b varies 14,236–14,242; c07c 17,263–17,275. Pose length matches C10095 for the first two inspected pose recordings, but c03b has 17,693 poses versus 17,685–17,690 video frames. Container starts of zero do not establish a common acquisition epoch. Unequal counts alone do not prove start offsets or locate dropped frames. [E4]

`visualize_single.py:108`, `render_batch.py:100`, `render_all_views.py:101` directly use decoded frame count as pose index, ignoring timestamps and per-view offsets. The toolkit README explicitly warns that original Assembly101 videos are not necessarily synchronized; its own visualizer is for a different synchronized rectified 30 fps release. Therefore identical raw frame indices cannot be assumed simultaneous. [E3, E10:110–113]

Official Assembly101 action annotations use a 30 fps extracted-frame convention, unlike 60 fps raw video/poses. The locally copied mistake README is absent, and its official repository describes interval timestamps without specifying a numerical time base. Local interval endpoints consistently fall near half the corresponding raw frame counts: 30 fps indexing is a strong inference for these files, but needs direct source confirmation/event matching. Do not silently apply a factor of two to every annotation product. [E4, S2–S3]

Missing evidence: per-camera acquisition timestamps/start offsets/drop maps, a documented pose-clock-to-video-PTS mapping, extracted-frame recording/frame/time manifest, and native contour-image frame identity. No static-window verification was found. Thus current eight-view packaging is not established as one physical state. [E4, E7]

### Exact follow-up validation

Select b06b, c07c, and c03b, preserving their raw-video and pose versions. For each camera, independently label visible hand landmarks and 3–5 sharp shared motion events in early/middle/late windows. Confirm joint names/hand identity first. Compare documented session-specific `K,D,R,t` hypotheses and estimated frame/time offsets using held-out events, reporting per-view median/95th-percentile pixel error, time residuals, and failure/visibility counts. Separate observed 2D labels from projections derived from the same 3D data. Examine PTS and synchronization residual drift to distinguish fixed offsets from dropped frames. Do not call camera self-consistency a ground-truth reprojection score. This test was not executed because independent labels and timing ground truth are absent.

## 6. Segmentation pipeline findings

**V:** SAM 2.1 loads `sam2.1_hiera_large.pt` (898,083,611 bytes) with `configs/sam2.1/sam2.1_hiera_l.yaml`. It uses `SAM2ImagePredictor.set_image/predict` independently for 88 JPEGs (11 per fixed camera), one fixed xyxy box per camera, no positive/negative points, no text, no temporal propagation, `multimask_output=False`. Boxes surround the truck/work area and sometimes adjacent tools; they are not whole-table masks. The bbox-check script disagrees with generation for C10395's y maximum (1100 versus 1050). Boxes depend on this scene/1920×1080 resolution. [E6]

No direct SAM 3 model loading or inference was found in project scripts or the adjacent SAM 2 clone. Eight files in `sam3_segmentations/` contain green boxes and magenta contours/tinted regions. Their name does not establish official SAM 3 version, checkpoint, prompts, execution location, or even model origin. Extracting masks from colors appears necessary because only rendered images were retained; why native model tensors were unavailable remains unknown. [E6]

The extractor selects magenta using HSV plus BGR tests, closes with a 7×7 kernel, selects the largest external contour above 300 px², fills it, then closes with 3×3. All eight saved masks reproduce **pixel-exactly** from the input visualizations. All are 1080×1920 uint8 with values `{0,255}`, one connected foreground component, areas 9,116–38,171 pixels (0.44–1.84% of the full image). This is validation of extraction reproducibility/binary format, not segmentation accuracy. [E6; `mask_diagnostics.json`, `extraction_diagnostics.json`]

Representative visual inspection: C10095, C10118, C10395, C10404 contour-derived overlays, C10118 original/clean packaged RGB, and C10118 deprecated SAM 2 frame 0005. The latter visibly includes a large shirt/background region and tool fragments. At frame 0005 across all cameras, deprecated masks contain 1–120 components; C10379 has a second component of 16,329 px. Contour-derived samples isolate the whole truck more cleanly and mostly exclude the adjacent screwdriver. C10118 has an internal opening in the source contour image; external-contour filling removes such holes. Boundary inflation/smoothing and potentially merged parts remain. No independent masks exist for IoU or boundary-accuracy claims. The eight still masks provide no temporal-quality evidence. [E6]

The target is visually a yellow crane truck with cabin, chassis, wheels, rotating upper structure and arm. The b06b labels independently describe attaching interior, wheels, arm, hook, turntable and cabin. Its exact source session is unproven, so that label correspondence is suggestive, not established. A whole assembled truck is not generically one rigid body through assembly/disassembly or articulation. A cabin/chassis subassembly during a verified rigid interval, or a separately manipulated rigid tool/component, is a more interpretable target. Very small symmetric wheels may be harder to track despite being rigid. [E4, E6]

## 7. Reconstruction pipeline findings

### Existing code, weights, and artifacts

Upstream SAM 3D Objects is at `f91db411c50efee93d8db7aeb323885650f6f722`. The two tracked modifications add configuration/constructor debug printing only: `inference_pipeline.py:271–275` and `sparse_structure_vae.py:347–352`. The model-parallel changes live in the untracked `custom_inference.py`, not a working upstream refactor. No inference traceback or saved project reconstruction dictionary/mesh/point cloud/preview/metric was found; `single_view/sam3d/output/` is empty. Vendored example assets are not Assembly101 results. [E8]

Actual configs/checkpoints include 6.69 GB SS-generator and 4.91 GB SLAT-generator checkpoints, decoders, encoders, and `pipeline.yaml`; external DINO and MoGe cache entries exist. Sizes establish presence, not checksum/integrity or successful load. `checkpoints/sam3d_objects_v1.pth` and `hf/ss_encoder.safetensors` are zero bytes; these are unusable placeholders, although the present pipeline references neither. See environment report for complete compact inventory. [E8, E12]

### Failure reconstruction and first-failure limits

1. `run_reconstruction.py:34–42` invokes `demo.py` with image/mask/output/export/no-Gaussian flags. **V:** `demo.py` has no argument parser and ignores those arguments, instead loading hardcoded upstream example assets and exporting a Gaussian splat to its current directory. The launcher cannot establish reconstruction of the selected project image even if the subprocess succeeds. It catches subprocess failure without propagating a failure status. [E8]
2. `custom_inference.py:9` imports the heavyweight inference module before patching OmegaConf. That module requires `CONDA_PREFIX` and imports compiled dependencies. **U:** historical import success is unrecorded. Current GPU access is already unusable; pipeline initialization calls `torch.cuda.current_device()` unconditionally at `inference_pipeline.py:102`. [E8–E9, E12]
3. **V:** `patched_load` removes keys `slat_decoder_gs`/`slat_decoder_gs_4`, while `pipeline.yaml` actually uses `*_config_path` and `*_ckpt_path`. It sets `device='cpu'` only if `device` exists, but the root YAML has no such key. The constructor default is `device='cuda'`. Both intended interventions are ineffective for this config. Both Gaussian decoders are initialized and default decode formats still include Gaussian. [E8:11–30, E9:70–75,147–155]
4. Hydra recursively instantiates `depth_model` before the pipeline; `DepthModel.__init__` defaults to CUDA and moves MoGe there. Then generators, decoders, and separate SS/SLAT conditioners are loaded onto the pipeline device, before any distribution step. The checkpoint loader reads `.ckpt` state on CPU, but `model.to(self.device)` makes the accumulated live models CUDA-resident. Each conditioner config has DINO image and mask encoders; the SS/SLAT conditioners are separately instantiated. Actual aliasing/peak memory was not measured. [E9:133–197,263–299,401–424; depth-model base:5–9]
5. Mesh decoder construction independently defaults `device='cuda'`. Its plain Python mesh extractor builds a dense grid at `resolution*4=256`. `256³×8` int64 cube indices alone are 1 GiB; `257³×3` int64 vertex coordinates add about 389 MiB, excluding construction temporaries, weights, activations and later mesh data. These allocations also precede distribution. This is a plausible OOM contributor, not proof it was the historical first failing allocation. [E9; mesh decoder:105–124; mesh utils:27–39]
6. If initialization completed, **V:** `pipeline.ss_condition_embedder` in custom line 68 does not exist in this pipeline; conditioners are in `pipeline.condition_embedders`. No compatible property was found. This is the first obvious post-initialization custom-code failure. [E8; E9:165–168]
7. If that were repaired, the wrapper only recurses tensors/dicts/lists. It leaves `torch.device` positional allocation arguments unchanged, ignores tuples and custom `SparseTensor`, and does not set a CUDA device context. `sample_sparse_structure` passes `image.device` to the generator, then sends its output to an unmoved SS decoder. SLAT conditioning is not assigned; the SLAT generator receives another device argument, and latent statistics/coords are handled on pipeline/image device. These are incompatible with the proposed independent GPU placements. [E8:32–51,68–71; E9:669–709,737–774]
8. `.to(cuda:3)` on the mesh decoder does not automatically move ordinary attributes inside non-Module `SparseFeatures2Mesh`/FlexiCubes. Its grid tensors and stored device remain tied to construction placement. The wrapper does not move its sparse latent input either. [E9; mesh extractor:54–66]
9. **V, isolated reproduction:** custom lines 74–78 pass PIL images; `Inference.merge_mask_to_rgba` calls `mask.astype` and NumPy slicing. Calling this exact AST-extracted method with two tiny PIL images raises `AttributeError: 'Image' object has no attribute 'astype'`. With ndarray mask `{0,255}`, multiplying uint8 by 255 yields alpha `{0,1}` through wraparound; upstream loader uses Boolean masks. The PIL error is independent of GPU capacity. [E8; notebook inference:94–108,344–355]
10. Mesh decoding returns a list of `MeshExtractResult`, not a `.save_obj` object. `custom_inference.py:84–87` catches the resulting AttributeError and only prints a notice, leaving no mesh. Upstream exposes a converted `glb` object. Moreover `postprocess_slat_output:550–553` accesses `outputs['gaussian'][0]` even with texture/postprocessing disabled: simply removing Gaussian decoding is insufficient for that path. [E8–E9]

**Conclusion:** OOM during concentrated initialization is plausible, with several allocations before distribution. The exact historical first exception, allocation size, device, and peak reserved/allocated VRAM are **U** because no log exists. No claim that the run reached any particular stage is warranted. Larger-memory hardware may support upstream single-device inference under its documented environment; it will not fix the custom script's attribute, type, device-transfer, and export defects. Four 11 GB devices do not provide one shared 44 GB allocation pool. [E8–E9, E12]

### Metric alignment

The package stores physical camera centers using calibration. **V:** the SAM 3D pointmap pipeline produces object rotation/translation/scale and a monocular-depth-derived pointmap; it does not jointly estimate a set of the eight physical camera centers in a common reconstruction frame. Single-view independent runs would not supply that set either. Rendering helpers generate synthetic orbit cameras; those are not calibrated observation centers. [E7; E9 pointmap:426–508; notebook inference:123–159]

**I, geometric conclusion:** Umeyama similarity fitting needs corresponding source and target 3D positions (at least three noncollinear pairs for a nondegenerate basic fit), such as the *same* recovered camera centers and their calibrated centers. Eight physical centers alone do not determine the unknown similarity of an arbitrary single-view object mesh. A single input camera center provides no baseline/scale constraint. The proposed alignment is therefore unjustified unless new correspondences/metric constraints are supplied.

Potential anchors: known-size object landmarks with multi-view correspondences; metric multiview depth; a known table plane plus sufficient object dimensions/orientation constraints; or an external metric mesh registered using calibrated views. When a metric object-to-camera transform exists, use a documented convention conversion and `T_world_object = inverse(T_camera_world) @ T_camera_object`. Monocular MoGe scale/shift and SAM predicted scale must not be labeled millimetres merely by applying calibrated extrinsics. The upstream `notebook/mesh_alignment.py` aligns a body mesh to MoGe by height/centroid; it is neither project Umeyama implementation nor independent metric validation. [E9; pose_target:107–128; mesh_alignment:124–143]

### Multi-view alternatives and visual hull

**V:** the multi-view directory is empty; no local DUSt3R, MASt3R, Pow3R, calibrated stereo, or visual-hull implementation/results were found. Suitability below is analytical, not execution evidence.

| Candidate | Potential role | Constraints for this subset |
|---|---|---|
| DUSt3R | Pairwise dense pointmaps and global scene alignment | Whole-scene background can dominate a 0.44–1.84% foreground object; unconstrained geometry needs metric anchoring; eight views imply up to 28 unordered/56 directed pairs |
| MASt3R | Matching and reconstruction with stronger correspondence information | Small low-texture plastic/symmetry/wide baselines remain difficult; metric predictions still require verification and frame registration |
| Pow3R | Use calibrated intrinsics/poses as explicit priors | Particularly relevant after calibration/timing verification; no local implementation/checkpoint/runtime measurement |
| Calibrated MVS/stereo | Recover metric visible surfaces without estimating known cameras | Require overlap/photometric consistency, correct rectification, matching at small object scale; occlusion, specular texture-poor parts and baseline angles are difficult |
| Visual hull | Calibration-based metric silhouette baseline | Uses masks instead of texture; no learned-model VRAM dependency; cannot recover hidden concavities and can overfill gaps |

Official DUSt3R/MASt3R/Pow3R sources support their stated interfaces; performance on these toy images is unknown. Crop-aware intrinsics must accompany object crops; avoid independently resizing views while keeping old K. Compare ROI reconstructions against full-scene geometry instead of assuming a model's global output represents the toy. [S5–S7]

Visual-hull feasibility is **conditional but credible**. One calibration file provides all eight cameras in a mathematically consistent candidate world; rotation checks pass and all views have nonempty masks. Physical common-world correctness and common object time/state are unvalidated. A numerical volume can be defined by backprojected silhouette cones and a conservatively chosen table-region bound, after establishing units and table geometry. Current files do not specify a trusted metric object AABB. [E5–E7]

Conceptual experiment only: establish recording/time identity; independently correct masks for a static rigid component; use session K/D/R/t with distortion-aware projection (or consistently undistort images/masks and use adjusted K); choose a conservative metric AABB; project voxel centers in chunks with positive-depth/in-image checks; retain silhouette-consistent voxels; compare strict intersection with a documented robust-view policy; extract a surface; evaluate silhouette reprojection on withheld views and manually measured dimensions. With an assumed 0.4 m cube, 2 mm spacing gives 200³=8 million voxels, eight views about 64 million projections, occupancy roughly 8 MB at one byte/voxel. At 1 mm: 64 million voxels and 512 million projections. These are arithmetic estimates, not benchmarks; float coordinate arrays/workspaces cost much more, so chunking matters. [I]

Limitations: hidden concavities, silhouette holes already lost, small fingers/arm structures, occlusion-driven false carving, background contamination, calibration/timing uncertainty, and absent underside views. Hand-occluded pixels should not be treated as trustworthy object absence. Report sensitivity to mask/calibration perturbations and label distances to the hull as approximate rather than true contact.

## 8. Tracking feasibility

**V/U:** no FoundationPose repository, checkpoint, build artifact, import, Assembly101 adapter, or execution result was found in the project/adjacent repository inventory. No FoundationPose package was identified in the two named environment metadata inspections. Installation elsewhere is unknown. Downloaded/installed/compiled/executed/integrated/validated are all **not established locally**. [E1, E12]

The official demo obtains depth each frame, calls `register(K,rgb,depth,ob_mask,...)`, then `track_one(rgb,depth,K,...)`. Translation initialization uses valid masked depth. Model-free setup does not imply an RGB-only observed-image pipeline. No suitable sensor depth is in the current local Assembly101 subset or its inspected stream inventory. [S4]

Conditional paths: calibrated multi-view stereo/learned depth could generate aligned metric depth and confidence, but must be validated for small parts before feeding it to a depth-dependent tracker. A framewise visual hull offers occupancy/motion without explicit pose. Known-mesh multiview silhouette optimization could estimate rigid pose using camera calibration and temporal priors. RGB feature/keypoint correspondence plus PnP is possible with a reliable model and discriminative texture, with multiview constraints reducing ambiguity. These alternatives require experiments; none is selected as final tracker.

Any evaluation must specify symmetry equivalences, visibility/occlusion policy, per-component identity, and rigid versus articulated intervals. Wheels/turntable/arm can move separately; one SE(3) cannot explain those motions. Initialize on a clear view/state, include multi-hypothesis/symmetry handling, and report track loss and reinitialization rather than smoothing through them. Record whether transforms map original mesh, centered mesh, camera, or world; apply millimetre-to-metre conversion explicitly if a tracker uses metres. Use symmetry-aware errors and multi-view reprojections; no pose ground truth presently supports ADD-style accuracy claims. [I; S4]

## 9. Downstream research feasibility

The subset supports a small investigation of event geometry: hand trajectories and semantic attach/detach/mistake/correction intervals exist. Part names are semantic labels, not persistent segmented instance IDs. There are no local contact labels, object-pose truth, procedural graph, or implemented recognition baseline. Mistake intervals provide some procedural boundaries but do not prove contact throughout an interval or a universal canonical action order. [E4]

| Feature | Minimum prerequisite | Full tracked 6D pose needed? |
|---|---|---|
| Hand speed, relative hands, trajectory/shape summaries | Correct hand schema, units, confidence and timestamps | No |
| Hand-to-object image distance / overlap | Aligned projected hands and per-frame masks | No; measures image interaction, not 3D contact |
| Hand-to-object metric proximity | Shared-world occupancy/point cloud/centroid, confidence | No; centroid proximity is coarser than surface distance |
| Fingertip-to-surface distance | Correct fingertip names and metric per-frame surface | Not if surface is reconstructed each time; pose needed to move a fixed mesh |
| Object translation/motion | Metric centroid trajectory or multiview tracked points | No for translation proxy |
| Motion correlation | Time-aligned hand/object trajectories | No for translational correlation |
| Relative object orientation / part articulation | Stable object axes/keypoints or tracked rigid/part pose | Usually yes for general orientation; symmetry remains |
| Persistent proximity / contact proxy | Time-aligned metric distances and validity handling | No in principle; proximity is not direct contact evidence |

Minimal evaluation proposal: choose several recordings/actors with repeated attach/detach and both correct/mistake examples, then manually verify a small set of interaction boundaries and rigid-part identities. Compare the same lightweight event model with (a) confidence-aware hand features, (b) hand plus 2D object features, (c) hand plus metric centroid/hull features. Match classifier capacity, time windows and label access; include object-only control and feature ablations. Split by entire recording and preferably actor/toy family; never randomly split adjacent frames or camera views of the same event. With only four actors and many toy IDs, treat leave-one-actor-out as a pilot with uncertainty, not a general benchmark claim. Report per-class precision/recall/F1, mistake PR-AUC/class support, boundary tolerance metrics, and geometric coverage/failure rates. Do not replace a missing metric estimate with zero and let missingness leak the class label. [I]

Relevant existing research baselines include HandFormer (hand-only and RGB fusion), MS-G3D/ISTA-Net as skeleton comparators, and the official Every Mistake Counts annotation benchmark. None has been reproduced locally. A metric-feature ablation should be evaluated against hand-plus-2D/RGB information so any improvement can reasonably be attributed to 3D object geometry. [S3, S8]

## 10. Compute and environment assessment

**V:** host identifies as `gpu2`, Ubuntu 24.04.3, kernel 6.14.0-33, Xeon W-2295, 18 cores/36 logical CPUs, about 125 GiB RAM and 115 GiB available, 8 GiB swap. Root/project filesystem is 879 GiB with about 564 GiB available. PCI inventory and NVIDIA proc entries show four RTX 2080 Ti devices. NVIDIA kernel module reports 580.65.06. `nvidia-smi` fails to communicate with the driver; both inspected PyTorch environments return `cuda.is_available() == False`, device count 0. Thus live GPU memory/capability/allocation status is unmeasured, despite hardware enumeration. No remote cluster inventory/authorized connection was available. [E12]

`fyp-pose`: Python 3.10.21; runtime torch 2.14.1+cu130, CUDA build 13.0, torchvision metadata 0.29.1, OpenCV runtime 5.0.0, NumPy 2.2.6. `sam3d-objects`: Python 3.11.0, torch 2.5.1+cu121, torchvision 0.20.1+cu121, NumPy 1.26.4/OpenCV 4.9.0.80 metadata, xformers 0.0.28.post3, spconv-cu121 2.3.8, Kaolin 0.17.0, PyTorch3D 0.7.8, FlashAttention 2.8.3. CUDA toolkit 12.1.105 is in the SAM 3D environment, not current PATH. Distribution metadata does not prove compiled extension import or execution. [E12]

RTX 2080 Ti is nominally an 11 GB Turing/sm75 GPU (hardware specification; usable VRAM was not read). SAM 3D's checked-in setup requires at least 32 GB VRAM. FlashAttention 2's official CUDA support targets Ampere/Ada/Hopper rather than Turing; installation alone is not compatibility. Local attention defaults to `sdpa`, so the FlashAttention issue is backend/path-dependent, not proof every SAM 3D run must invoke it. CUDA 13-era packages in `fyp-pose` merit a separate Turing compatibility check; do not infer support from a successful CPU import. [E9 attention config, E12, S9–S10]

Detailed checkpoint sizes, environment distinctions, and candidate-specific bottlenecks are in [ENVIRONMENT_AUDIT.md](ENVIRONMENT_AUDIT.md). No heavyweight benchmark or compiled extension suite was launched.

## 11. Major technical risks

**P0:** unvalidated metric/timing correspondence; unresolved hand schema/validity; arbitrary single-view metric alignment; whole-articulated-object rigid assumption; no experiment/evaluation proving the research contribution.

**P1:** broken SAM 3D custom path and live GPU access; missing observed depth for FoundationPose; no temporally identified object-mask sequence. **P2:** untracked/ignored critical code and data provenance, no locked environment, stale paths, lossy contour masks, unsafe success/default behavior. **P3:** misleading all-400/multi-GPU status messages. Each risk has traceable evidence and an action in [ARCHITECTURE_RISKS.md](ARCHITECTURE_RISKS.md).

## 12. Recommended next experiments

1. **Establish one trustworthy clip.** Recover or re-establish source recording/frame/time for eight views. Choose a rigid part and a clearly static state plus a short manipulation interval. Confirm joint ordering, units, hand identity, and invalid-value policy. Deliver a manifest and independent timing/reprojection measurements before reconstruction claims.
2. **Measure segmentation.** Annotate a small stratified set of clear/occluded views and motion frames; compare native SAM 2 predictions and any traceable SAM 3 output using IoU/boundary error, identity continuity, and failure counts. Preserve model tensors, prompts and native masks. Do not grade an extractor against its own input contour as object accuracy.
3. **Build a calibrated visual-hull baseline in a future implementation phase.** Compare withheld-view silhouettes, known dimensions, strict versus robust intersections and calibration/mask perturbations. Success requires nontrivial geometry consistent across views, not merely a nonempty mesh.
4. **Run a small event-feature ablation.** Evaluate hand-only versus hand-plus-2D versus hand-plus-metric object features on recording/actor-held-out intervals. Start with proximity/motion; reserve claims about physical contact and full orientation until supported.
5. **Test learned reconstruction conditionally.** After restoring GPU access, the supervisor may choose an unmodified upstream SAM 3D control on a supported ≥32 GB GPU and/or crop-aware Pow3R/MVS comparison using the same validated views. Log exact first failure and peak VRAM per stage; validate mesh scale/frame against the geometric baseline. Do not invest first in the current four-way wrapper.
6. **Select a tracker only after geometry/depth tests.** Compare a short rigid interval with calibrated multiview silhouette/keypoint optimization; evaluate FoundationPose only if aligned depth quality is adequate. Include occlusion, symmetry and reinitialization cases.

These are proposed experiments, not executed development. The audit stops here.

## 13. Evidence references

All local paths are relative to the project root unless absolute. Line references describe the audit snapshot; diagnostic JSONs contain numerical evidence, not implementation additions.

| ID | Primary local evidence |
|---|---|
| E1 | Root and vendor `git --no-optional-locks status`, `rev-parse`, `log`, remotes, `ls-files`; root `.gitignore`, `README.md`; `inspection_record.md` |
| E2 | `scripts/download_dataset.py:8–73` |
| E3 | `scripts/visualize_single.py:6–51,53–79,89–119`; `scripts/render_batch.py:8–44,69–117`; `scripts/render_all_views.py:9–48,72–113,120–149`; notebook code/output cells |
| E4 | `data/raw_videos/recordings/`, `data/poses/poses@60fps/`, `mistake_repo/annots/`; `dataset_inventory.json`, `pose_diagnostics.json`, `decode_sample.json` |
| E5 | `assemblyhands-toolkit/calib/nimble_json_calib/*.json`, specifically c07c calibration; `calibration_diagnostics.json` |
| E6 | `experiments/sam2_test/generate_masks.py:19–93`, `check_bboxes.py:5–15`; `scripts/extract_sam3_masks.py:8–73,117–134`; experiment images; `mask_diagnostics.json`, `extraction_diagnostics.json` |
| E7 | `scripts/package_reconstruction_data.py:12–15,46–71,104–161`; `experiments/reconstruction_data/session_01/metadata.json` and RGB/mask/RGBA folders |
| E8 | `3D_reconstruction/single_view/sam3d/run_reconstruction.py:12,34–56`; `sam3d_repo/custom_inference.py:9–30,32–51,62–87`; upstream `demo.py:9–21`; vendor diff |
| E9 | Under `sam3d_repo`: `notebook/inference.py:83–119,123–159,344–355`; `checkpoints/hf/pipeline.yaml`; `sam3d_objects/pipeline/inference_pipeline.py:70–205,263–424,543–574,651–774`; `inference_pipeline_pointmap.py:262–315,400–508`; `pipeline/depth_models/base.py:5–9`; mesh decoder/extractor/grid files discussed in section 7 |
| E10 | `assemblyhands-toolkit/README.md:27–85,110–113`; `src/dataset/__init__.py:8–80`; `src/dataset/AssemblyHands-Ego/dataset.py:49–56,122–147`; `visualization/video_visualizer.py:25–90` |
| E11 | Empty `3D_reconstruction/multi_view/`, empty project `single_view/sam3d/output/`; selective artifact/source-name searches |
| E12 | `/etc/os-release`, `uname`, `lscpu`, `free`, `df`, `lspci`, NVIDIA proc/version files, `nvidia-smi`, two Python runtime probes and conda/dist-info metadata; `ENVIRONMENT_AUDIT.md` |
| E13 | Mathematical feasibility analysis in sections 7–9, explicitly conditional rather than executed |

Official references consulted on audit date:

- S1: [Assembly101 downloader and annotation products](https://github.com/assembly-101/assembly101-download-scripts).
- S2: [Assembly101 annotation frame-rate convention](https://github.com/assembly-101/assembly101-annotations).
- S3: [Official mistake annotation schema and benchmark](https://github.com/assembly-101/assembly101-mistake-detection).
- S4: [FoundationPose official demo](https://github.com/NVlabs/FoundationPose/blob/main/run_demo.py) and [estimator/depth use](https://github.com/NVlabs/FoundationPose/blob/main/estimater.py).
- S5: [DUSt3R official implementation](https://github.com/naver/dust3r).
- S6: [MASt3R official implementation](https://github.com/naver/mast3r).
- S7: [Pow3R paper](https://arxiv.org/abs/2503.17316) and [NAVER camera/scene-prior overview](https://europe.naverlabs.com/research/3d-foundation-models/).
- S8: [HandFormer official implementation](https://github.com/s-shamil/HandFormer).
- S9: [SAM 3D checked-in setup requirements](../../3D_reconstruction/single_view/sam3d_repo/doc/setup.md) (local authoritative snapshot).
- S10: [FlashAttention official supported hardware](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support).
- S11: [HandFormer cleaned-pose reader, getPose3d](https://github.com/s-shamil/HandFormer/blob/main/Preprocess/1_get_clip_data.py), hand-key mapping and wrist index 5; fine-grained annotation-to-60-fps conversion is also explicit in its header.

## 14. Unresolved questions

See [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md). The critical external evidence is the extracted-frame/SAM 3 provenance, acquisition timing/calibration records, original OOM traceback, available supported compute, and the supervisor's precise experimental endpoint. Unknowns are explicit limits on conclusions, not permission to assume the architecture is validated.
