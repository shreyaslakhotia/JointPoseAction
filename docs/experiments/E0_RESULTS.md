# E0 results — no trained-model results yet

Unattended continuation is prepared in [E0_HANDOFF.md](E0_HANDOFF.md), attaching
to preparation job 242798 (last user-reported running). One gated GPU sanity/baseline
job and dependent final CPU evaluation/report will be submitted by the user-run
helper. Local synthetic/mocked validation passes 48 tests. No real baseline has
yet been submitted or evaluated at this repository checkpoint.

Preparation checkpoint after plan job 242776: the selected 335 recordings have
all companions, requiring 7.994 GB transfer / 23.098 GB extracted. Candidate
100-class counts are 28,194 / 9,060 / 13,256. A resumable CPU acquisition and
preprocessing job is prepared. E0 uses dimensionless hand-length-normalized
features and declared reference-reader timing with cadence/boundary exclusions;
physical metric units and independently verified RGB clocks are not asserted.
Training readiness now awaits the real retained corpus and support checks.

Current decision after successful companion job 242754: exclude recordings 9031
and 9026 instead of attempting recovery; use source hand-confidence scores to mask
zero-score hands, preserving zero coordinates when scores are positive. Per-recording
cadence and segment-boundary checks are implemented. Final corpus coverage and
feature-unit/reference-time documentation remain pending; no training has run.

Complete-sample checkpoint: CPU job 242734 completed `0:0`, verifying all three
reported pose members. Two sampled recordings have severe annotation/pose coverage
mismatches under the nominal 30/60 mapping (22/28 and 302/325 segments outside
pose bounds). Separate timestamp/confidence/transform files exist and need
inspection. Prior inventory-based class/sample counts do not yet describe a
validated training corpus. See the complete-sample findings in the schema report.

Current checkpoint, 2 November 2026: GPU smoke 241925 passed on L40S; authenticated
metadata inspection 242218 passed; cached feasibility job 242713 completed `0:0`
in 14 seconds with all 28 CPU tests passing in 2.585 seconds. The 100-class candidate
has 28,457 / 9,060 / 13,256 train / validation / test segments before pose-quality
exclusions. Units/reference-frame, validity conventions and full-recording temporal
alignment still block training. See [E0_SCHEMA_VALIDATION.md](E0_SCHEMA_VALIDATION.md).
The historical engineering snapshot below is superseded where it differs.

No E0 training or real-data evaluation has run. Top-1, Top-5, verb/noun accuracy,
F1, confusion matrices and learning curves are **not measured**. No checkpoint
or performance figure is presented as a result.

Executed engineering evidence: 24 local CPU unit/synthetic tests passed, public
source revision/file-size inspection succeeded, and the unauthenticated small
HF file probe returned HTTP 401. Dataset payloads and cleaned samples have not
been acquired locally. Environment installation job 241924 completed `0:0`
according to user-supplied accounting. GPU smoke job 241925 was queued at the
latest supplied status; its successful execution has not been reported.

Real training readiness: **blocked** by authenticated CSV acquisition, actual
cleaned pose sample/schema/units and time-origin/endpoint validation, measured
recording/class/split coverage, reviewed selected pose transfer sizes, production
preprocessing and cluster CPU/GPU validation. The code/configuration/interfaces
are prepared, but that is not completion of E0.

After a later authorized run, record subset scope, class/sample counts, dataset
hash, source revisions, train/validation/test protocol, exclusions, checkpoint
selection, job/exit status, metrics against majority/chance, missingness-related
errors and overfitting behavior. Results must be labeled subset results unless
the full official protocol is followed. Fusion suitability is not yet established.
