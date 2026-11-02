# Data validation — no new pilot sample acquired

No pilot video, cleaned pose sample or calibration sample has been downloaded
or validated during Phase 1A. Historical artifact hash checks establish preserved
bytes, not current pilot provenance or semantic/metric accuracy. No representative
package or overlay has been generated for this phase.

| Acceptance check | Current result |
|---|---|
| Video files/headers/frame counts/decodes | Not tested on cluster |
| Recording vs action-label membership | Official format located; actual rows pending |
| Per-recording calibration and camera model | Not validated |
| Pose schema, identity, units, confidence | Source differences documented; sample pending |
| Cross-camera synchronization | Unresolved |
| Action 30 FPS to video/pose time mapping | Unresolved |
| Diagnostic overlays/reprojection residuals | Not produced |
| Reproducible RGB/pose/calibration/label package | Not produced |

For an approved sample, log source revision and hashes, camera file metadata and
PTS, selected decoded frames, original pose fields, joint-name mapping, missing
values, validity and exact action rows. Save calibration with transform direction,
units, distortion and image dimensions. Document offsets/clock origins per view;
`time = action_frame / 30` is a nominal reference, not independent evidence of
capture synchronization. `raw_frame = 2 * action_frame` is not an established
mapping. Endpoint inclusion is also pending.

Use visible events in early/middle/late windows to check offsets/drift; retain
unresolved timing as uncertainty. Save compact overlays and diagnostics; plausible
overlays do not prove millimetre accuracy. Do not reuse an unrelated session's
calibration or turn unknown mask provenance into verified truth.

After validation, replace pending entries with actual filenames, extraction
commands, sample counts, residuals and limitations. This report deliberately
does not claim completion of the original sample-validation acceptance criteria.
