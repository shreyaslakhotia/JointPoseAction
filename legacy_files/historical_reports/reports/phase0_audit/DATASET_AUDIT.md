# Dataset audit — 1 November 2026

Root: `/home/mlda/shreyas_projects/JointPoseAction`. Read-only probes; no downloads or full-video decoding. V = directly verified, I = inferred, U = unknown. See the main report for evidence references and scientific implications.

## Downloaded recording inventory

All 20 directories are under `data/raw_videos/recordings/`; every row has eight fixed RGB views, a matching `data/poses/poses@60fps/<session>.json`, and `mistake_repo/annots/<session>.csv`. All 160 video headers report 1920×1080, average and nominal 60/1 fps, and stream start time 0. These are encoded-stream properties, not proof of physical synchronization. No HMC video or sensor-depth stream is present in this local subset.

Camera IDs, present in every recording: `C10095`, `C10115`, `C10118`, `C10119`, `C10379`, `C10390`, `C10395`, `C10404`, each suffixed `_rgb.mp4`. Toolkit mapping is exo1 through exo8 in that order (`assemblyhands-toolkit/src/dataset/__init__.py:55–63`). All four expected ego views are absent locally; they are outside the downloader's C*.mp4 selection.

The table lists minimum–maximum across the eight views. Exact per-view metadata and file sizes are in [dataset_inventory.json](dataset_inventory.json). Split entries are **AssemblyHands toolkit membership only**, not Assembly101 action-benchmark membership.

| Recording ID | Frames across views | Duration seconds | Correct / mistake / correction intervals | Toolkit split |
|---|---:|---:|---:|---|
| `nusar-2021_action_both_9011-b06b_9011_user_id_2021-02-01_154253` | 14,236–14,242 | 237.267–237.367 | 6 / 0 / 0 | Not established |
| `nusar-2021_action_both_9011-b08c_9011_user_id_2021-02-01_154736` | 23,367–23,377 | 389.450–389.617 | 9 / 0 / 0 | Not established |
| `nusar-2021_action_both_9011-c01c_9011_user_id_2021-02-01_155620` | 20,396–20,406 | 339.933–340.100 | 11 / 1 / 1 | Not established |
| `nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239` | 13,818–13,822 | 230.300–230.367 | 8 / 0 / 0 | Not established |
| `nusar-2021_action_both_9011-c13b_9011_user_id_2021-02-01_160915` | 21,266–21,277 | 354.433–354.617 | 9 / 4 / 4 | Not established |
| `nusar-2021_action_both_9012-a16_9012_user_id_2021-02-01_162904` | 26,119–26,129 | 435.317–435.483 | 11 / 2 / 0 | Not established |
| `nusar-2021_action_both_9012-a17_9012_user_id_2021-02-01_162209` | 21,520–21,537 | 358.667–358.950 | 9 / 2 / 2 | Not established |
| `nusar-2021_action_both_9012-b06d_9012_user_id_2021-02-01_163713` | 16,783–16,788 | 279.717–279.800 | 6 / 0 / 0 | Not established |
| `nusar-2021_action_both_9012-c06d_9012_user_id_2021-02-18_121034` | 17,684–17,685 | 294.733–294.750 | 8 / 2 / 2 | Not established |
| `nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345` | 17,263–17,275 | 287.717–287.917 | 6 / 0 / 0 | val |
| `nusar-2021_action_both_9012-c14b_9012_user_id_2021-02-01_164944` | 28,459–28,472 | 474.317–474.533 | 11 / 3 / 0 | Not established |
| `nusar-2021_action_both_9013-a02_9013_user_id_2021-02-02_130807` | 34,547–34,551 | 575.783–575.850 | 8 / 5 / 2 | Not established |
| `nusar-2021_action_both_9013-a28_9013_user_id_2021-02-02_134923` | 13,936–13,937 | 232.267–232.283 | 7 / 2 / 1 | Not established |
| `nusar-2021_action_both_9013-b01a_9013_user_id_2021-02-02_135446` | 24,865–24,867 | 414.417–414.450 | 9 / 1 / 0 | Not established |
| `nusar-2021_action_both_9013-c03b_9013_user_id_2021-02-24_113410` | 17,685–17,690 | 294.750–294.833 | 6 / 0 / 0 | Not established |
| `nusar-2021_action_both_9013-c09c_9013_user_id_2021-02-24_113951` | 30,574–30,578 | 509.567–509.633 | 16 / 11 / 4 | Not established |
| `nusar-2021_action_both_9014-a12_9014_user_id_2021-02-02_141945` | 24,644–24,647 | 410.733–410.783 | 3 / 0 / 0 | Not established |
| `nusar-2021_action_both_9014-a23_9014_user_id_2021-02-02_142800` | 26,387–26,390 | 439.783–439.833 | 8 / 0 / 0 | train |
| `nusar-2021_action_both_9014-b02a_9014_user_id_2021-02-02_143628` | 20,125–20,126 | 335.417–335.433 | 4 / 0 / 0 | Not established |
| `nusar-2021_action_both_9014-b05a_9014_user_id_2021-02-02_144248` | 57,279–57,283 | 954.650–954.717 | 14 / 6 / 0 | Not established |

V: twenty sessions span four actor IDs (9011–9014) and 20 session-specific toy IDs. Vehicle category names are not supplied in a local object taxonomy; do not infer exact category from filename letters. Semantic part names and attach/detach verbs are available in the CSVs. The visually inspected packaged object is a yellow crane truck, but its source recording is not recorded.

V: video files total 175,955,481,912 bytes (~163.87 GiB). There are 87 pose JSONs totaling 5,626,354,608 bytes (~5.24 GiB): 20 match these recordings and 67 have no matching local recording directory. There are 328 mistake CSVs. A count of 100 in downloader configuration is a requested batch size, not the actual number of downloaded video recordings.

## Integrity and corruption limits

All 160 ffprobe calls returned readable stream metadata without reported probe errors. The audit decoded first/middle/last frames of C10095 and C10118 for b06b, c07c and c03b: 18/18 succeeded, dimensions 1080×1920×3. See [decode_sample.json](decode_sample.json) for full filenames/indices. No corruption was detected in this limited sample; the rest of each bitstream was not decoded or checksummed, so full-file integrity and acquisition frame drops remain unknown.

Small probe command (read-only; replace the path with any manifest video):

```bash
ffprobe -v error -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,start_time -of json <video.mp4>
```

## Pose source and schema

V: project download and renderer paths use original Assembly101 `poses@60fps`. No cleaned AssemblyPoses archive or refined AssemblyHands annotation data was found. The official downloader distinguishes legacy versus cleaned poses ([source](https://github.com/assembly-101/assembly101-download-scripts)). A toolkit clone and its calibration examples are not evidence of using its refined pose release.

V: legacy JSON is a top-level list; rows have integer `frame_index`, floating `timestamp`, `landmarks` keyed `0`/`1`, and per-hand `tracking_confidence`. Representative arrays are 21×3 xyz. These files have no local accompanying schema naming every joint or defining units/hand IDs. Current scripts call 0 left and 1 right and draw MediaPipe-like consecutive chains, but do not validate those choices. They ignore `tracking_confidence`, missing/zero joint semantics and timestamp.

| Complete JSON inspected | Rows / index range | Timestamp range | Uniformly sampled rows | Zero xyz joints / confidence-zero hands in sample |
|---|---|---|---:|---:|
| `nusar-2021_action_both_9011-b06b_9011_user_id_2021-02-01_154253.json` | 14,242; 0–14241 | 8298.766–8536.116 | 201 | 3 / 3 |
| `nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345.json` | 17,275; 0–17274 | 11951.566–12239.466 | 201 | 1 / 1 |
| `nusar-2021_action_both_9013-c03b_9013_user_id_2021-02-24_113410.json` | 17,693; 0–17692 | 79726.699–80021.566 | 202 | 3 / 3 |

All three complete files have consecutive frame indices and both hand keys on all rows. Sampled xyz contain no NaN/Inf, but that does not make low-confidence poses valid. Confidence ranges are approximately 0–0.98; increments in timestamp are ~0.016–0.017 s. Sampling is deterministic: `rows[::max(1,len(rows)//200)]`; count finite/zero values and hand confidences on those rows. [pose_diagnostics.json](pose_diagnostics.json) records sample sizes and projections. Only three of 87 files were parsed completely, not the entire pose collection.

I: units appear to be millimetres, based on hand coordinates and calibration translations; world-to-camera use gives plausible positive depths. U: authoritative physical units, global gravity/axes, legacy joint-name mapping, and hand-ID semantics. Resolve from annotation producer documentation plus visible landmark checks. Toolkit refined ordering is right 0–20, left 21–41, wrists 20/41 (`README.md:70–85`, dataset loader:49–56); this is a different schema and must not be applied without mapping.

## Calibration inventory and correctness limits

V: `assemblyhands-toolkit/calib/nimble_json_calib` contains 20 small calibration JSONs. Only two exact session filenames overlap the 20 local recording folders: c07c (9012, 2021-02-01 164345) and a23 (9014, 2021-02-02 142800). The project uses c07c for all current custom rendering/packaging. No separately downloaded `data/calibrations` directory exists.

V: fixed camera records specify `DistortionModel=OpenCV`, 1920×1080 and nested 4×4 `ModelViewMatrix`. K is fx/fy/cx/cy; OpenCV distortion order is k1,k2,p1,p2,k3 (remaining fixed-camera coefficients zero). Current direct `R=E[:3,:3]`, `t=E[:3,3]` implements `X_camera=R X_world+t`; camera centers `-R.T@t` follow from that convention. No transpose is used. The four ego records use OVFishEye62 and are not covered by this projection implementation.

V: eleven unique records per fixed camera across the 20 calibration files. Against c07c, maximum matrix differences are 707–975 translation-scale units and focal differences up to 4.14 pixels. Variation could reflect world redefinition, rig changes or recalibration; the exact cause is U. It disproves using the word universal without a recording mapping. Within the c07c file, rotation determinants and orthogonality pass floating-point consistency checks. See [calibration_diagnostics.json](calibration_diagnostics.json).

V: pose sample projections using c07c have 100% positive depths and 96.75–100% inside-image fractions. This supports one algebraic convention, but cannot establish camera accuracy, calibration applicability, units, anatomical correctness or synchronization. No independent 2D ground truth, measured pixel reprojection error, or project validation test was found.

## Synchronization and annotation times

V: every session has unequal frame counts across cameras. a17 has the largest 17-frame spread (~0.283 s at 60 fps). The two inspected Feb-1 pose lengths match C10095; c03b has 17,693 poses, exceeding all view lengths (17,685–17,690). A common zero container start does not reveal recording offsets. The code uses video count == pose frame index without capture-clock mapping or dropped-frame correction.

The local toolkit explicitly warns original Assembly101 videos are not necessarily synchronized (`assemblyhands-toolkit/README.md:110–113`). Its own 30 fps rectified-video visualizer is for another video release (`visualization/video_visualizer.py:25–60`); that code is not evidence that the raw 60 fps inputs here are synchronized.

[Official action annotations](https://github.com/assembly-101/assembly101-annotations) use frames extracted at 30 fps. [Mistake annotation documentation](https://github.com/assembly-101/assembly101-mistake-detection) identifies interval timestamps but does not state their numeric time base. The copied local mistake directory has no README. Endpoints near half the raw frame count support a 30 fps interpretation for the local mistake files, but it remains I until checked against source/events. Inclusive/exclusive endpoint conventions are also U. Do not equate pose timestamps with elapsed clip time without subtracting/validating a recording origin.

Required validation: annotate several sharp events/visible landmarks in early, middle and late windows per camera; compare acquisition offsets/drift against pose time and video PTS, reserve events for testing, and report pixel and time residuals. Metadata alone cannot establish dropped-frame positions. No expensive frame scan was performed.

## Masks, RGB and reconstruction package

V: 88 RGB JPEGs (11 per camera, camera-only sequence names), 88 deprecated SAM2 masks and 88 overlays, eight claimed SAM3 visualizations, eight contour-derived binary masks plus eight overlays, and eight RGB/mask/RGBA package triples. `metadata.json` names each source as `CAMERA_0005.jpg`; every packaged RGB is pixel-identical to the decoded named JPEG. Names are not original recording/frame IDs. Mask and RGB are selected independently in packager lines 58–71.

| Camera | Foreground px | Percent full image | Foreground components | Binary values |
|---|---:|---:|---:|---|
| C10095 | 15,161 | 0.731% | 1 | 0, 255 |
| C10115 | 9,116 | 0.440% | 1 | 0, 255 |
| C10118 | 38,171 | 1.841% | 1 | 0, 255 |
| C10119 | 13,734 | 0.662% | 1 | 0, 255 |
| C10379 | 20,796 | 1.003% | 1 | 0, 255 |
| C10390 | 9,955 | 0.480% | 1 | 0, 255 |
| C10395 | 14,925 | 0.720% | 1 | 0, 255 |
| C10404 | 9,757 | 0.471% | 1 | 0, 255 |

All eight masks have dimensions 1920×1080; exact color-extractor re-execution in memory reproduces each stored mask. Native segmentation correctness was not tested against independent annotations. See [mask_diagnostics.json](mask_diagnostics.json) and [extraction_diagnostics.json](extraction_diagnostics.json). Largest-mask ranking chooses C10118 (38,171 px), which is area ranking, not a measured reconstruction-quality score.

V visual examples: deprecated SAM2 C10118_0005 includes shirt/background and adjacent-tool fragments; its mask has 12 components and 53,291 foreground pixels. At the same saved sample index across views, counts range 1–120 components. Contour-derived C10118/10095/10395/10404 overlays more cleanly outline the truck. Source C10118 contour includes an internal hole, while external contour filling removes holes. The extraction can merge boundaries/lose disconnected components by design. RGB-camera appearance and area differences do not by themselves establish cross-view identity/state consistency.

U: exact source session, physical timestamp of each mask and RGB, native SAM3 model/checkpoint/prompt, and whether all eight views correspond to an identical rigid state. The toy has movable arm/turntable/wheels; whole-toy rigid tracking across the procedure is not justified.

## Procedural annotations and coverage

V: 224 intervals across the 20 sessions: 169 correct, 39 mistake, 16 correction. Each row contains start/end, verb, two working part names, label, optional remark. All 20 have no nonpositive duration intervals; this check does not validate semantic labels or event alignment. Attach/detach boundaries are annotation intervals, not contact ground truth. Part names do not supply 3D instance correspondences or geometry.

Potential pilot: choose repeated parts/actions and both error/non-error intervals; manually verify a small representative set; evaluate recording/actor-held-out hand-only, hand+2D and hand+metric-proximity features. Count annotation support and report failures; do not claim a general benchmark from this lexicographically selected four-actor subset. The downloaded videos were selected from sorted mistake filenames, not a documented random or balanced sample.

## Remaining data gates

Before metric fusion: recover frame/time provenance; confirm legacy hand naming/units/validity; obtain a valid calibration assignment per selected recording; establish cross-camera and pose/video timing; verify mistake time base and split; select a rigid component/state. Full corruption checks, cleaned/refined annotation comparison, independent reprojection errors and ground-truth IoU remain unperformed for explicitly stated input/evidence limits.
