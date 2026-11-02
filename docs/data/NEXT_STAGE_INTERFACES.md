# Minimal next-stage input/output contracts

These are proposed contracts, not dataset converters, models or verified schema
mappings. Preserve original annotations alongside normalized fields.

## Recognition example

One record per physical action event: stable `segment_id`, exact `recording_id`,
source dataset/revision, official `split`, original action/verb/noun IDs and names,
original interval endpoints/timebase, endpoint convention and camera/view source.
Deduplicate equivalent view rows while preserving their provenance.

Arrays: `timestamps_seconds[T]`, `hand_xyz[T,2,21,3]`,
`hand_valid[T,2,21]` and optional `hand_confidence[T,2,21]`. Store explicit hand
order, joint names, source schema, physical units and coordinate-frame identifier.
Canonical hand order may be `[right,left]` only after verified source mapping.
Preserve missingness; a missing hand must not become a valid zero-coordinate hand.
Do not invent confidence when a source provides only validity.

Temporal metadata: source timestamps/frame IDs, recording origin, resampling rate,
resampling method and valid temporal support. Preserve unresolved correspondence;
do not fill gaps silently. Keep official recording membership and avoid leakage
from overlapping events or simultaneous views. An optional `object_features`
field may be absent for E0; future features must include timestamps/support,
object identity, frame/units and feature provenance. No network architecture is
implied by these arrays.

## First metric reconstruction

Inputs: exact recording ID and calibrated RGB frames with camera IDs, source
frame/PTS, exposure/synchronization uncertainty, image dimensions, camera model,
intrinsics, distortion and explicitly directed extrinsics; verified object masks
with object/component identity, original prompts/manual edits and model/source
revision. All views must refer to the same static object configuration or a
documented temporal window. Check mask alignment and provenance before use.

Outputs: point cloud or occupancy artifact, object ID, coordinate-frame definition,
physical units, world transform if applicable, source view/time support, method
version/configuration and calibration/mask provenance. Include coverage and
reprojection/silhouette consistency diagnostics, and scale checks where independent
reference exists. Do not label silhouette consistency a metric-accuracy measure.

## Minimum hand–object integration requirements

Static geometry describes shape but does not establish the object's position at
every action frame. Temporally localized geometry can support an event/window
if its validity interval and shared frame are known. Full 6D tracking provides
time-varying rigid transforms but is not assumed necessary for the first baseline.
Geometry and hands need compatible units, coordinate transforms, temporal support,
object identity and validity. Shape-only features can be frame invariant; spatial
hand–object distances require verified placement at that time. The supervisor
chooses representation/experiment design after the pilot evidence is available.
