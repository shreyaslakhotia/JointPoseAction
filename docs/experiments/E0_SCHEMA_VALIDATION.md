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
7,789,211,135 bytes and unpacked 22,415,512,050 bytes. This does not establish which
337 overlap the 361 annotated recordings. The full archive was not acquired;
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
