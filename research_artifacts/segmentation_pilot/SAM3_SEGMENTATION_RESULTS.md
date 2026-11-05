# SAM 3 diagnostic segmentation: execution record

**Status: both jobs completed, outputs inspected; reconstruction readiness remains
false.** See [the completed inspection](INSPECTION_270796.md) for measured results,
before/after figures and concrete failures. The preparation details below remain
the reproducibility record, not a claim of independent segmentation accuracy.

## Model and inputs

- Official implementation: `facebookresearch/sam3`, commit
  `f1647e8d0c1d280991f4d1b7d974c2704537fe66`.
- Checkpoint: `facebook/sam3`, revision
  `3c879f39826c281e95690f02c7821c4de09afae7`, file `sam3.pt`.
- Bytes: `3450062241`; SHA-256:
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
- Access approved and checkpoint downloaded/verified by CPU job `269584`.
- Inputs: 24 native initial RGB frames, three physical toys x eight cameras,
  independently inferred with the single text prompt **`toy truck`**.
- A second stage adds each predeclared positive box to the same text prompt.
  Box coordinates and the agent intervention are recorded per image. No action
  label, drawn mask, instance union or silhouette completion is used.
- Native source RGB remains 1920 x 1080. SAM's internal image preprocessing is
  the official processor; saved masks return to native source dimensions.

## Completed jobs and artifacts

| Job | Stage | Request | Dependency |
|---|---|---|---|
| 270796 | Official pretrained SAM 3 inference | one A6000 GPU; 45-minute limit | none |
| 271689 | Native-mask decoding, descriptive tables and review figures | 4 CPU cores; 30-minute limit | `afterok:270796` |

No training, fine-tuning, extra reconstruction model or real reconstruction is
part of either job. The model runs with bfloat16 autocast, inference mode and
compilation disabled. Only one checkpoint is stored. The source checkout and
checkpoint hashes are checked before model import.

Each image keeps `text_only/` and `refined/` independently. Each stage saves
all returned positive-score query masks in lossless, packed native binary form,
with candidate IDs, boxes, scores, foreground area and image coverage. The
storage threshold is 0; the descriptive detection threshold is **score > 0.5**.
Neither count is measured correct-object recall. The processor's field named
`masks_logits` contains sigmoid probabilities; it is not treated as raw logits.
Optional full probability storage is disabled to respect quota.

The CPU renderer produces eight-camera originals, binary-mask sheets,
transparent overlays, identical-region prompt comparisons, a per-view CSV and
an empty review-decision template. Its highest-score query display convention
does **not** identify the correct physical object. Full candidates remain
available for reviewing wrong-instance cases, including printed toy pictures.

## Metrics and interpretation

| Quantity | Current result |
|---|---|
| Correct physical object segmentation success | Agent qualitative inspection complete; user approval and independent accuracy unavailable |
| Confidence / mask coverage / predicted instance counts | 48 per-view/stage rows saved; see inspection report |
| Text-only versus box-refined change | 19/24 to 24/24 views with a score > 0.5; background/tool inclusion prevents interpreting this as accuracy gain |
| Cross-camera object-instance consistency | Toys generally located; silhouette quality varies and user review remains pending |
| IoU, Dice, boundary F1 against independent reference masks | Unavailable; no independent reference masks |
| Accepted reconstruction-ready pilots | **0** |

No prediction is used as its own ground truth. Accuracy remains unestablished.
The user will inspect successes, ambiguities, occlusions and failures before
any subsequent real reconstruction is authorized. Calibration accuracy,
synchronization and metric units remain unresolved separately from mask quality.
All manifests and candidate outputs retain `diagnostic_2d_review=true` and
`multiview_dataset_ready=false`; completion cannot promote either scientific gate.

## Validation and reproducibility

The final combined local suite collected **214 tests: 212 passed and two
OpenCV-dependent tests were skipped**. It includes 48 reconstruction-preparation
tests. The cluster job reruns all pilot tests with OpenCV before model execution. The renderer test
checks that native holes and separate stages survive exactly and that a high
score never becomes ground-truth acceptance. Earlier cluster geometry tests
passed 45/45, and local-stability tests passed 6/6.

An initial environment setup hit two measured dependency conflicts:
`contourpy` required NumPy 2 and AnyIO required newer `typing_extensions`.
Overlay pins `contourpy==1.3.3` and `typing_extensions==4.16.0` resolved them.
CPU NumPy/Torch transfer, torchvision NMS, SAM processor import and `pip check`
then passed; the recognition base inventory remained unchanged.

Source: [`sam3_runner.py`](../../segmentation_pilot/sam3_runner.py),
[`review.py`](../../segmentation_pilot/review.py),
[`requirements`](../../reproducibility/environments/sam3-pilot-requirements.txt).
The run is staged outside the cluster Git checkout so local repository work
cannot alter the batch script already submitted to Slurm. Complete file paths
are in [pilot selection](PILOT_SELECTION.md).

Inspect without rerunning:

```bash
sacct -X -j 270796,271689 --format=JobID,JobName,State,ExitCode,Elapsed -P
squeue --start -j 270796,271689
tail -n 60 /projects/jointposeaction/logs/seg-sam3-270796.err
```

For a deliberately new reproduction, stage the pinned source and configs under
the dedicated project directory, follow the environment/discovery/survey scripts,
and submit `scripts/cluster/segmentation_sam3.sbatch`. Its diagnostic manifest
requires exact hashed survey/calibration/tracking evidence and native frames;
it cannot silently fall back from a failed validated pilot. Use a new job output
directory, retain native predictions, and attach the CPU review script with
`--dependency=afterok:NEW_JOB --kill-on-invalid-dep=yes` and
`--export=ALL,JPA_SEGMENTATION_JOB=NEW_JOB`. Do not submit a duplicate merely to
inspect this queued run.
