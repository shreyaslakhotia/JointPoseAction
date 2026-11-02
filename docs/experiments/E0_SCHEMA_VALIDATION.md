# E0 source/schema checkpoint — 2 November 2026

## Action targets are provided by Assembly101

The [official paper, section 3.3](https://assembly-101.github.io/assets/Assembly101.pdf)
describes action labels with start/end times and overlapping fine-grained events.
The [annotation format](https://github.com/assembly-101/assembly101-annotations/tree/main/fine-grained-annotations)
defines actions.csv and train/validation/test segment CSVs with verb, noun and
action IDs/names. These human-annotated action targets are distinct from the
automatically generated hand trajectories. AssemblyHands is not the source of
our E0 fine-grained recognition labels; historical mistake CSVs are not substituted.

CPU job 242218 completed `0:0`, parsed the pinned official CSVs and passed 26 tests
in the installed cluster environment. User-supplied coverage: 1,380 class mappings;
47,651 / 15,699 / 21,934 canonical train/validation/test segments across 211 / 62 /
88 recordings. No recordings overlap. 47 participant IDs appear across splits.
928,239 alternate-view rows were accounted for rather than independent pose-only
examples. Overlapping same-recording segments remain distinct examples.

## Protocol decision

The user explicitly selected official recording-disjoint splits. Preprocessing
now defaults to `report_official_overlap`; participant overlap is documented and
is not misrepresented as participant-independent generalization. Recording overlap
remains a hard failure. Class vocabulary still uses training counts only; neither
validation nor test performance can select classes, architecture or checkpoints.

The candidate analysis considers all pose-available official recordings, with
up to 100 action classes having at least 50 training events. These are proposed
training-only thresholds, not a finalized vocabulary or measured valid-pose count.
No arbitrary 20-recording legacy subset is used. Coverage/transfer sizes must be
computed before deciding the corpus; preprocessing must recheck class counts after
quality exclusions. Test data will not be used for model or hyperparameter selection.

## Actual cleaned sample evidence and limits

The inspected cleaned archive has 337 landmarks3D JSON members: total compressed
7,789,211,135 bytes and unpacked 22,415,512,050 bytes. CPU job 242713 established
that all 337 overlap official annotated recordings: 196 train, 58 validation,
83 test. The full archive was not acquired;
bounded inspection transferred 691,850 bytes.

The first three sampled JSON entries have frame keys 1000/1001/1002 and two 21x3
arrays. They are a partial object prefix, not proof of global index origin or
minimum frame. The frame-1000 right wrist is exactly zero while other joints are
nonzero. This is compatible with a reference origin and is not proof of missingness.
Without producer validity fields we cannot convert zeros to invalid observations.

The adapter preserves finite zero coordinates when the zero-invalid rule is off,
records zero observations as ambiguity, and distinguishes finite numeric availability
from tracking confidence. The example real profile now preserves zeros. An actual
zero-as-missing rule requires authoritative evidence; no such rule is approved.
There is no invented confidence value. Fully absent frames and nonfinite values
remain explicitly unavailable. A valid-zero-wrist regression test covers both
hand articulation and wrist availability.

The official paper places the 21 tracked landmarks in world coordinates. The
reference cleaned reader assigns hand keys 0/1 to left/right and wrist to index 5.
The producer-family [UmeTrack landmark enumeration](https://github.com/facebookresearch/UmeTrack/blob/main/lib/common/hand.py)
also has tips 0–4, wrist 5, finger frames 6–19, palm center 20. This corroborates
the reader topology but does not by itself certify every cleaned release field.
Physical units, exact cleaned reference frame, confidence/absent-observation
conventions and complete time origin/endpoints remain unverified. Coordinate
magnitudes resembling millimetres do not establish a metric conversion.

The nominal 30 FPS annotation / 60 FPS pose formula is documented by the released
reader, including two pose offsets for each annotation frame. Independent tests
cover that formula; actual representative recording endpoints/clock origins still
need examination. No arbitrary shift is introduced based on prefix key 1000.

## Next CPU evidence and training gate

`e0_feasibility.sbatch` analyzes only cached inspection/CSV/sample files. It writes
exact intersection coverage, missing pose IDs, proposed training-only vocabulary,
class support, participant/toy diversity, member transfer sizes and partial-sample
wrist/zero diagnostics. It does not download or convert poses, train or request GPUs.

Training remains blocked until the pose convention and time alignment are verified
without a materially ambiguous feature transformation. The user authorized a tiny
sanity/overfit run and one single-GPU baseline once those checks pass; no additional
permission question is needed for those already-authorized runs. No run has been
submitted at this checkpoint, and no result is claimed.

## Feasibility result: CPU job 242713

User-supplied logs establish `COMPLETED`, `0:0`, 14 seconds, with all 28 CPU
unit/synthetic tests passing in 2.585 seconds. The saved detailed report is
`/projects/jointposeaction/data/annotations/e0-feasibility-242713.json`.

| Official split | Pose-available recordings | Pose-available segments | Candidate segments, 100 classes |
|---|---:|---:|---:|
| Train | 196 / 211 | 45,029 / 47,651 | 28,457 |
| Validation | 58 / 62 | 14,926 / 15,699 | 9,060 |
| Test | 83 / 88 | 20,954 / 21,934 | 13,256 |

The candidate contains 50,773 segments. Vocabulary selection used training counts
only (minimum 50 training examples, maximum 100 classes); these counts precede
pose-quality and temporal-boundary exclusions. Per-class evaluation support is
in the detailed report and must be reviewed before finalizing the corpus. Official
recording-disjoint membership is retained; participant-independent performance
must not be claimed.

All 337 cleaned members are needed for this candidate: 7.789 GB compressed or
22.416 GB extracted. Raw float32 features have a 2.808 GB upper bound before file
overheads; labels, indexes, checkpoints, caches and the existing environment need
additional space. These are decimal GB. Do not retain both a large archive and
the extracted corpus without measuring dedicated-project free quota first.
No full pose member or video was downloaded by the feasibility job.

Training toy names are all the placeholder `-`; this is missing metadata, not
one toy category. Validation/test list 14/15 named categories. Do not infer train
toy diversity from this field or use evaluation categories to select classes.

The right wrist is zero in frames 1000, 1001 and 1002; wrist-to-index-proximal
distance is approximately 99.1133 source units for both hands. These diagnostics
do not verify units or distinguish a valid origin from a stale/default pose.
In particular, three frames with zero wrist position do not prove missingness.
Preserve zeros, do not invent tracking confidence, and inspect complete training
recordings with companion transforms before validating motion features.

Training remains blocked on physical/reference-frame semantics, missing/stale pose
conventions and real recording origin/endpoint checks. The next bounded acquisition
must report member sizes first and read complete members with ZIP CRC verification.
No new feasibility sweep or GPU reservation is needed.

The subsequent user-supplied summary confirms all 100 selected classes occur in
every split. Minimum class support is 96 train / 9 validation / 17 test; only one
validation class has fewer than ten examples. Candidate participant counts are
47 / 35 / 40. This supports retaining the training-selected vocabulary, with
per-class uncertainty acknowledged and support rechecked after quality exclusions.

`e0_validate_samples.sbatch` prepares the next CPU inspection. The three smallest
training members total exactly 4,938,281 compressed bytes and 15,757,192 extracted
bytes, as reported before acquisition. Hard-coded size checks refuse a changed
plan; HTTP ranges are capped at 8,000,000 bytes including ZIP directory/header
overhead, with no full-archive fallback. Each complete member is read through EOF
for ZIP CRC verification and hashed. Companion files are inventoried only. The
report compares candidate base/endpoint mappings without automatically selecting
one or changing the production profile. It counts zero wrists and exact stationary
runs as observations, never as invented confidence or invalidity labels. This
inspection has not executed at the time of this repository update.

First submission 242731 failed before output-directory creation or remote pose
access: the sample script supplied only train.csv to the shared loader, which
requires all three official split paths. The correction verifies all CSV hashes,
loads all official splits, rechecks recording leakage, and then retains training
segments only for sample diagnostics. A synthetic integration regression calls
this exact annotation-loading helper with the real three-split loader and checks
that evaluation rows are not returned and changed hashes are refused. The archive
transfer plan and training gate are unchanged.

## Complete-sample findings: job 242734

The corrected CPU job completed `0:0` in 2m34s with empty stderr. All three pose
members were read completely, CRC-verified and SHA-256 hashed. Actual HTTP range
traffic was 5,597,651 bytes. No video, full pose archive or GPU was used.

| Training recording suffix | Pose frame IDs | Annotated segments | Outside nominal zero-base, inclusive mapping |
|---|---|---:|---:|
| 9031, 2021-02-04_101221 | 0 through 1386 | 28 | 22 |
| 9086, 2021-02-16_160311 | 0 through 1771 | 14 | 0 |
| 9026, 2021-02-03_163855 | 2 through 2956 | 325 | 302 |

Each observed ID range is internally contiguous. The prefix key 1000 was not
the recording start. Changing annotation index base by one or switching endpoint
inclusivity does not resolve the large coverage mismatch in recordings 9031/9026.
Do not shift, stretch, rescale time or clamp these segments to fit the poses.
Possible causes such as partial trajectories or a distinct timestamp convention
remain hypotheses, not conclusions. The smallest-member selection is biased toward
short files; these three recordings do not estimate corpus-wide exclusion rates.
The 50,773 candidate examples and their class support remain unvalidated counts.

All coordinate values are finite and no hand is entirely zero, yet recording 9031
has zero wrists in 970/1387 left frames and 553/1387 right frames, with exact
stationary whole-hand runs of 155/69 frames. Recording 9026 has stationary runs of
77/42 frames. Finite numeric availability is therefore insufficient to establish
tracking validity. Zeros remain preserved; no threshold or default-pose mask is
introduced on this evidence alone.

The complete ZIP inventory revealed separate `hand_confidences`, `timestamp` and
`xf_transf` members for all three recordings. Earlier statements about absent
confidence apply only to the landmarks3D JSON schema, not to the whole release.
These companions must be inspected before deciding feature masks or time alignment.
Their total is exactly 1,335,648 compressed bytes / 3,621,046 extracted bytes.
`e0_inspect_companions.sbatch` acquires only those nine members with a 3,000,000-byte
HTTP range budget, complete CRC/hash checks and no full-archive fallback. It reports
actual structures/examples, timestamp increments, pose-key intersection, scalar
confidence/zero-wrist associations if that schema is observed, and transform
translation/wrist residuals if 4x4 hand-keyed matrices are observed. These are
diagnostics, not inferred units, thresholds or synchronization guarantees.

Training remains blocked. No production pose profile has been marked validated,
and no actual recognition result is available. The companion job is prepared but
has not yet executed at this checkpoint.
