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
