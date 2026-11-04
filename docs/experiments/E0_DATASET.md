# E0 dataset - frozen recognition source contract

> **Current consolidated status (4 November 2026):** E0 and both Phase 1C
> recognition runs are complete. The [final comparison](../experiments/DUAL_BACKBONE_COMPARISON.md)
> records the measured results and frozen population (22,083 train / 6,880
> validation / 9,760 test segments; 100 classes). The phase log below preserves
> its original observations; statements about pending acquisition, validation
> or training describe earlier stages, not current readiness. Use the
> [environment guide](../cluster/ENVIRONMENT_SETUP.md) for setup and the final
> experiment reports for current outcomes. Absolute video/pose synchronization
> and metric hand/object calibration remain unverified.

Latest user evidence: cluster job 242218 parsed all official label metadata,
passed 26 tests, inventoried 337 cleaned pose members and sampled three actual
frames. The official recording-disjoint protocol is now selected; participant
overlap is reported. Finite zero coordinates are preserved. See
`E0_SCHEMA_VALIDATION.md` for current measurements and unresolved semantic checks;
earlier planning defaults below are superseded by that checkpoint.

Reviewed 2 November 2026. No real E0 examples have been preprocessed. This report
distinguishes public inventory, released reader behavior and actual file validation.

## Source evidence and acquisition

The public HF API pins `cvml-nus/assembly101` at
`bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967`. The fine-grained directory lists:

| File | Bytes |
|---|---:|
| `actions.csv` | 81,530 |
| `head_actions.txt` | 3,139 |
| `train.csv` | 100,677,017 |
| `validation.csv` | 34,544,546 |
| `test.csv` | 47,993,475 |
| Total approved metadata | **183,299,707** |
| `AssemblyPoses.zip` (not downloaded) | **71,972,006,749** |

The user approved the 183.30 MB metadata acquisition directly onto dedicated
cluster SSD. No such data were downloaded locally. A bounded request for the
7.8 KB annotation README returned HTTP 401 without authentication; approval in
the user's account does not supply a token to the agent. Public listing access
does not establish asset-download access. No credentials were requested/read
from unrelated projects. Some upstream LFS digests are masked in the public API;
they are not usable checksums. Local source evidence is in
`reproducibility/datasets/e0_source_evidence.json`.

`scripts/data/inspect_e0_sources.py` authenticates using project-configured HF
credentials, checks pinned file sizes, acquires only the approved CSV/mapping
files, hashes actual contents and computes split coverage. It also permits
at most 4 MB total archive ranges, no single range above 2 MB, plus a 32 KB
decompressed cleaned-pose prefix. It requires exact HTTP 206/Content-Range;
there is no full-archive fallback. A large ZIP central directory may exceed the
bound; that failure is retained for review rather than escalating the download.
No full pose recording or video acquisition is implemented by this command.

## What the authoritative documentation establishes

The [official fine-grained format](https://github.com/assembly-101/assembly101-annotations/tree/main/fine-grained-annotations)
defines the action/verb/noun mapping and explicit train/validation/test CSVs.
Intervals use 30 FPS; the documented vocabulary has 1,380 actions. Actual CSV
headers, row counts, per-split recordings/participants, distributions and overlap
counts await authenticated payload inspection. No historical 20-recording subset
or AssemblyHands split is substituted for the recognition splits.

## Cleaned adapter evidence and unresolved fields

The official [HF distribution](https://huggingface.co/datasets/cvml-nus/assembly101)
recommends cleaned `AssemblyPoses.zip` rather than legacy `poses@60fps`.
The released [HandFormer reader](https://github.com/s-shamil/HandFormer/blob/640a10a6bd03f85842a2662e67cf30444ab1d673/Preprocess/1_get_clip_data.py)
consumes this archive: `assembly101_camera_and_hand_poses/landmarks3D/RECORDING.json`.
The implemented adapter accepts that reader's frame-keyed object, with hand keys
`"0"`/`"1"` containing 21x3 xyz arrays. It preserves source order `[left,right]`;
the reader identifies wrist index 5. This is source-code evidence, not a claim
that a downloaded sample was verified. Legacy list-of-frame dictionaries and
AssemblyHands 42-joint ordering are rejected instead of guessed or auto-detected.

Full joint names, physical units, coordinate-frame axes, missing/zero semantics,
confidence availability and actual time origin still require release/sample
evidence. The reader exposes no confidence field in the cleaned hand arrays;
the adapter reports confidence unavailable instead of inventing it. A reviewed
profile must supply explicit source joint order, units conversion, frame identity,
validity policy, origin/offset and endpoint semantics. Unknown example fields in
`configs/e0_source_manifest.example.json` are deliberately unset. Real preprocessing
refuses an unvalidated contract. Synthetic tests cannot masquerade as real runs.

The [official MS-G3D graph](https://github.com/assembly-101/assembly101-action-recognition/blob/00d3c3bb3af75ce38ae406614a7cb034aecdebc0/MS-G3D-action-recognition/MS_G3D/graph/assembly101_hands.py)
also identifies wrist-centered topology at index 5. We do not infer a complete
cleaned joint-name table from that legacy-oriented code.

## Temporal and segment conventions

HandFormer's cleaned-pose preprocessing explicitly doubles annotation indices
and reads both 60 FPS offsets for each 30 FPS annotation frame. Its segment loop
supports inclusive end frames. For zero-based indices and zero offset, the
candidate mapping is `pose_indices = [2*start, 2*(end+1))`. The implementation
requires explicit index origin, frame offset and endpoint policy instead of
silently fixing those assumptions. Tests cover inclusive/exclusive endpoints,
nonzero origins/offsets, gaps and coverage bounds. `frame_index / 60` is a nominal
pose time; no capture-clock or cross-camera synchronization claim follows from
it. Actual sample/annotation endpoint correspondence remains unverified.

Each canonical-view CSV row is a distinct segment example. Overlapping intervals,
including identical intervals with distinct annotation IDs, remain separate.
There is no framewise single-label action array. As in the reference reader,
only `C10119_rgb.mp4` label rows are used to avoid multiplying pose-only events
across views; every other row is counted as `alternate_view`. No RGB is loaded.
Duplicate canonical annotation IDs fail validation. Label triples must agree
with the official mapping; no filename-derived labels.

## Preprocessing and splits

All source paths/hashes and selected recording split/participant identities are
explicit. Source integrity and official recording overlap are checked before
selection. The default `report_official_overlap` policy preserves official
recording-disjoint splits and reports shared participants. Do not describe this
as participant-disjoint.

Select classes by canonical training-event counts only: default minimum 50,
maximum 100, frequency descending with ID tie-break, then contiguous labels ordered
by official action ID. No validation/test counts enter selection. Every omitted
event receives a reason. If quality exclusions empty a split/class or lower its
training count below the configured minimum, preprocessing fails and its saved
artifact remains not ready for training. No test example is silently removed.

Sampling: 64 deterministic nearest-frame positions per segment; no interpolation,
smoothing, rotation normalization or learned statistics. Missing frames and
nonfinite coordinates are masked. Source hand-confidence scores must be positive;
finite zero coordinates alone are never a missingness test. Wrist-relative joint coordinates plus both wrist trajectories
relative to the first available segment origin preserve inter-hand placement
and motion. Features contain 126 local xyz, 6 wrist xyz, 42 raw-valid and 42
local-valid indicators: **216 features per time step**. The production E0 profile
divides coordinates by the median positive-score wrist-to-index-proximal length
within the segment. These dimensionless features are invariant to a common unknown
source-unit multiplier; they are not labelled metres. Physical calibration remains
necessary for metric E3 fusion. Source/frame metadata remains available for
future geometric features. Default minimum sampled local validity is 25%; it is
an engineering threshold requiring review on actual data, not a confidence claim.

Prepared artifacts store per-segment source identity, labels, frame/time mapping,
quality, feature hash, split summaries and complete exclusions. No output directory
is overwritten. Acquisition plan 242776 selects 335 candidate recordings and 100
training-selected classes; final counts await acquisition and quality exclusions.
