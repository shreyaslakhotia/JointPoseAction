# Pilot plan — selection pending metadata access

> **Current consolidated status (4 November 2026):** E0 and both Phase 1C
> recognition runs are complete. The [final comparison](../experiments/DUAL_BACKBONE_COMPARISON.md)
> records the measured results and frozen population (22,083 train / 6,880
> validation / 9,760 test segments; 100 classes). The phase log below preserves
> its original observations; statements about pending acquisition, validation
> or training describe earlier stages, not current readiness. Use the
> [environment guide](../cluster/ENVIRONMENT_SETUP.md) for setup and the final
> experiment reports for current outcomes. Absolute video/pose synchronization
> and metric hand/object calibration remain unverified.

Status: no exact session/interval selection is approved or download-ready.
We have not retrieved the official split CSV payloads or inspected cleaned poses.
Inventing concrete rows would contradict the requirement to verify identities,
coverage, calibration and storage sizes. The historical 20 sessions are not
automatically candidates or the final benchmark.

## Bounded selection targets

Pilot A: a software proof of concept using approximately 30–60 distinct physical
action events, 5–10 labels where feasible, and at least two recordings. Use
official training membership for development; include a separate official
validation recording if shared class coverage allows. Do not tune on test data.
These are engineering sampling targets, not a new benchmark design or a claim
that this small sample measures recognition quality.

Pilot B: one rigid toy/component, at least three calibrated fixed RGB views,
and a 1–3 second static observation window with limited occlusion. Verify object
identity and static behavior visually before accepting the window. It may share
a Pilot A session if all requirements hold. Inspect a later manipulation interval
to clarify temporal limitations; do not infer full 6D tracking is required.

Actual recording IDs, action intervals, camera IDs and bytes remain pending:

| Field | Required selection evidence |
|---|---|
| Recording / split | Actual official split row and source revision |
| Time interval | Original annotation frame endpoints/timebase and verified mapping |
| Pose source | Sample schema, validity, units, hand identity and temporal coverage |
| Camera views | File inventory and matching per-recording calibration |
| Rigid/static object | Decoded visible evidence, mask source and initialization quality |
| Transfer size | Pinned source file sizes, including full MP4/pose files if required |

## Storage control

50 GB SSD is sufficient to begin. Cap initial source-video acquisition at 10 GB,
selected annotations/poses at 5 GB and derived pilot material at 5 GB. Preserve
10 GB overall headroom after accounting for the environment and caches. These
are ceilings, not expected measured sizes or permission to download that volume.
HDD allocation is unnecessary now; revisit only after a real archival need.

Do not extract every frame of every recording. Decode selected frames/windows;
include extraction rate, dimensions, encoding and output-size estimate before
running. No pilot videos will be downloaded until exact rows and total transfer,
disk use and temporary overhead have been reviewed and approved.

`reproducibility/datasets/phase1a_pilot.json` records this blocked selection state.
It contains no executable download requests. Fill it from actual evidence rather
than allowing a downloader to choose substitute sessions or calibration.
