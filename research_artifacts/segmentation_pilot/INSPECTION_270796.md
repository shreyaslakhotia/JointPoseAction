# Completed SAM 3 pilot inspection

Inspected 5 November 2026. **The diagnostic segmentation and visual-review jobs
completed as intended. The broader reconstruction-ready multiview dataset stage
is not complete. No real reconstruction is authorized or performed.**

## Execution and artifact integrity

| Job | Result | Hardware | Elapsed |
|---|---|---|---|
| 270796, SAM 3 inference | COMPLETED, 0:0 | NVIDIA RTX A6000 | 6 min 3 s |
| 271689, CPU review figures | COMPLETED, 0:0 | cpu-2 | 24 s |

All **24 native frames** completed both stages: three physical toys, eight fixed
cameras, text-only and positive-box refinement. All **9,600 returned query masks**
were retained: 200 queries per frame per stage. This is not a count of detected
objects. The model stayed pinned to the recorded official source/checkpoint.
Model/inference elapsed time was **329.70 seconds**, and peak allocated CUDA
memory was **8.53 GiB**. The full job duration also includes preflight/import work.
The cluster pilot suite passed **75/75**, including its OpenCV tests.

The review job produced **18 figures and 48 per-view/stage measurement rows**.
All 18 retrieved figure hashes match its manifest, and the inference completion
and provenance hashes match. All **48 displayed native binary masks** were checked
for native dimensions, binary encoding, foreground count, bounding box and
selected-candidate confidence against their candidate metadata; no mismatch was
found. The CPU renderer checked every packed native-mask file's SHA-256 before
rendering. JPEG masks/overlays are previews, never authoritative mask data.

The displayed query is the highest score in each stage. Agent visual inspection
is recorded separately; the original predictions and pending-user-review flags
remain unchanged. No alternate query was selected, mask edited, artifact repaired
or new inference submitted during this inspection.

## Descriptive results

Counts below mean **views with at least one query score > 0.5**, not measured
segmentation recall or success. Scores are model confidence, not accuracy.

| Physical target | Text-only qualifying views | Box-refined qualifying views | Mean highest score, text / refined |
|---|---:|---:|---:|
| green01, garbage truck C07C | 7/8 | 8/8 | 0.796 / 0.940 |
| crane02, crane truck B05B | 4/8 | 8/8 | 0.498 / 0.934 |
| green03, garbage truck C08B | 8/8 | 8/8 | 0.877 / 0.940 |

Overall this is **19/24 to 24/24 views above the score threshold**. Higher scores
did not consistently produce cleaner silhouettes. The highest-score refined
mask is larger than its text-only counterpart in every view. Mean per-view area
ratios are 1.158, 1.447 and 1.277 respectively. Area growth can include recovered
parts, shadows/background or a separate tool; it is not an accuracy improvement.
Some stages also choose different query IDs, so these are displayed-output
comparisons, not tracking of a fixed query identity.

## Per-object visual assessment

### green01: physical toy located, occlusion and tool leakage remain

Text-only predictions are generally tighter in the unobstructed views. Box
refinement adds table/background around C10095, C10115 and C10390. More seriously,
the refined **C10119 and C10404 masks include part of the yellow screwdriver**,
which is explicitly outside the target object. In C10379, instruction paper hides
much of the truck; the mask covers visible upper surfaces, not an unobstructed
whole-object silhouette. That view must not be treated as a complete silhouette
for carving. Existing C10118 local-stability failure is not overridden.

[Original / text / refined comparison](results-270796/green_white_garbage_truck_c07c_prompt_comparison.jpg)
and [eight-camera refined overlay](results-270796/green_white_garbage_truck_c07c_refined_overlay.jpg).

### crane02: prompting is sensitive and refinement adds background

The text prompt gives scores below 0.5 in C10095, C10115, C10390 and C10404,
although the low-score displayed queries still overlap the physical toy.
Box refinement raises confidence in all four. It also includes substantial
surrounding table in **C10115, C10390 and C10404**, and the support surface beneath
the toy in C10379. Fine boom/hook parts and holes require closer boundary review.
Articulation and cross-view static configuration remain separate unresolved
checks. This object cannot be released on confidence alone.

[Original / text / refined comparison](results-270796/yellow_red_crane_truck_b05b_prompt_comparison.jpg)
and [eight-camera refined overlay](results-270796/yellow_red_crane_truck_b05b_refined_overlay.jpg).

### green03: strongest visibility candidate, boundaries still need approval

The physical truck is visible across all eight initial views, and all text-only
views exceed the score threshold. Some text-only masks omit portions of dark
wheels. Refined outputs often add table/background, especially C10095 and C10115;
**C10395 adds a spurious protrusion above the toy**. This is a promising candidate
for subsequent per-view mask review, not an accepted reconstruction pilot.

[Original / text / refined comparison](results-270796/green_white_garbage_truck_c08b_prompt_comparison.jpg)
and [eight-camera refined overlay](results-270796/green_white_garbage_truck_c08b_refined_overlay.jpg).

## Completion assessment and remaining gates

| Requirement | Assessment |
|---|---|
| Three candidate toys, all eight source views and provenance | Complete as a diagnostic survey; only two object families |
| Pinned official SAM 3, separate native text/refined outputs | Complete |
| Review figures, transparent native overlays, descriptive tables | Complete; user mask review pending |
| Correct physical instance and suitable silhouette for every usable view | Not accepted; background/tool inclusion, omissions and occlusion remain |
| Calibration assessment with actual source/image evidence | Performed; geometric accuracy remains inconclusive |
| Temporal correspondence or separately verified static-state scope | Not passed |
| Coordinate conventions and physical scale | Mathematical convention documented; physical units unverified |
| Independent segmentation accuracy | Unavailable; no independent reference masks |
| Reconstruction-ready accepted pilots | **0** |

IoU, Dice and boundary F1 are intentionally unavailable. The model's own masks
cannot establish those accuracy scores. Mask review cannot replace camera
accuracy or timing/static-state evidence. Neither successful Slurm completion
nor the scores above changes the closed real-reconstruction gate.

The next work, only after further instruction, would be per-view mask selection
or targeted correction plus resolution of the camera/correspondence evidence.
No automatic refinement promotion or reconstruction run follows this report.

## Files and storage

Lightweight review figures and measured metadata are in
[`results-270796/`](results-270796/). Its
[native inspection record](results-270796/native_mask_inspection.json) includes
all per-view measurements, candidate identifiers and checks. The original
[per-view CSV](results-270796/per_view_descriptive.csv),
[completion record](results-270796/completed.json),
[provenance](results-270796/provenance.json) and
[figure manifest](results-270796/review_manifest.json) are preserved.

Complete native outputs remain on the dedicated cluster SSD:

- `/projects/jointposeaction/experiments/segmentation_pilot/sam3-270796/`
  contains full-resolution originals, every packed binary query mask and its
  candidate metadata. Measured bytes: **71,347,969**.
- `/projects/jointposeaction/experiments/segmentation_pilot/review-271689/`
  contains the review package and `native_display_examples/` with native binary
  PNGs and transparent RGBA overlays. Measured bytes: **20,924,615**.
- Logs: `/projects/jointposeaction/logs/seg-sam3-270796.{out,err}` and
  `seg-review-271689.{out,err}`.

Ceph project usage at inspection: **39,806,342,842 / 50,000,000,000 bytes**,
leaving **10,193,657,158 bytes**. No new model, video archive, environment,
GPU/CPU job, reconstruction or training was started for this inspection.

RGB attribution and derivative-figure notes are in the
[research-artifact index](../README.md). The figure overlays are native prediction
visualizations; no cosmetic cleanup was applied to conceal failures.
