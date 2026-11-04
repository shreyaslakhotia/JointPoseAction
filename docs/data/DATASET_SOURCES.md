# Dataset sources and schema — 2 November 2026

> **Current consolidated status (4 November 2026):** E0 and both Phase 1C
> recognition runs are complete. The [final comparison](../experiments/DUAL_BACKBONE_COMPARISON.md)
> records the measured results and frozen population (22,083 train / 6,880
> validation / 9,760 test segments; 100 classes). The phase log below preserves
> its original observations; statements about pending acquisition, validation
> or training describe earlier stages, not current readiness. Use the
> [environment guide](../cluster/ENVIRONMENT_SETUP.md) for setup and the final
> experiment reports for current outcomes. Absolute video/pose synchronization
> and metric hand/object calibration remain unverified.

This is a documentation review, not evidence of dataset file acquisition.
Source pages were reviewed live. File revisions/hashes and schemas requiring
sample contents remain unverified. Existing Phase 0 findings are dated evidence.

Update: public HF API inspection pinned the revision and exact official metadata
sizes; unauthenticated small-file retrieval returned HTTP 401. The user approved
183.30 MB official CSV/mapping acquisition on the cluster. The released cleaned
reader contract and temporal formula were inspected at pinned source revisions.
See `docs/experiments/E0_DATASET.md` for current evidence, implementation and
unresolved sample/units/timing checks. No full pose archive or video was downloaded.

## Official distribution and access

The [Assembly101 project](https://assembly-101.github.io/) links its
[Hugging Face dataset](https://huggingface.co/datasets/cvml-nus/assembly101) and
announces migration from Drive in May 2026. The HF page currently requires login
and acceptance of access conditions. Public documentation is readable; that does
not prove the user's account can download files. Accept conditions in the user's
browser and authenticate privately on the cluster when acquisition is needed.
Never paste a token or log credential files. Dataset license: CC BY-NC 4.0;
retain attribution and document derived artifacts.

The HF card lists `recordings/`, `poses@60fps/`, `AssemblyPoses.zip` and
`annotations/`. The [official downloader documentation](https://github.com/assembly-101/assembly101-download-scripts)
also describes selective recordings/views and distinguishes cleaned poses from
legacy poses. Its Drive route has separate access requirements. Do not use
the old project downloader's `/blob/` ZIP URL or default 100-recording selection.

No full dataset, feature archive or model weight download is required here.
The current exact HF revision, file sizes, checksums and access success have not
been queried with an authenticated client. The old cached legacy revision is
`bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967`; it is historical evidence, not a pin
for the current pilot. Resolve a revision before listing/downloading pilot files.

## Recognition labels and splits located

The [annotation repository](https://github.com/assembly-101/assembly101-annotations)
and its [fine-grained format documentation](https://github.com/assembly-101/assembly101-annotations/tree/main/fine-grained-annotations)
identify `actions.csv`, `train.csv`, `validation.csv`, `test.csv` and
`head_actions.txt`. The CSV data are linked through Drive; the inspected GitHub
directory itself exposes the format README, not those CSV payloads. The HF card
also advertises an annotations folder; exact filenames there await gated listing.
The recognition [benchmark repository](https://github.com/assembly-101/assembly101-action-recognition)
is a reference, not an installation dependency for this phase.

The format defines 1,380 actions from 24 verbs and 90 nouns. `actions.csv` maps
action/verb/noun IDs to names. Split CSVs contain recording/view paths,
`start_frame`, `end_frame`, action/verb/noun IDs/names, toy ID/name and shared-toy
and RGB indicators. The frame reference is **30 FPS**, while raw videos are
60 FPS. Endpoint inclusion and exact source-view synchronization need validation.
The deprecated challenge test CSV is not the current evaluation split.

We have located authoritative definitions but have not acquired or enumerated
their recording membership. Preserve the official splits when constructing the
pilot; group simultaneous views of an action as one physical event. Repeated
view rows must not silently become independent hand-only examples. Do not infer
action split membership from AssemblyHands splits or old mistake CSVs.

## Pose alternatives

| Product | Known evidence | Appropriate use / remaining uncertainty |
|---|---|---|
| Legacy `poses@60fps` | Old audit found per-frame index/timestamp, 21x3 landmarks per hand and tracking confidence | Reproduce historical inputs only; hand IDs, joint names and units not authoritatively established |
| Cleaned Assembly101 in `AssemblyPoses.zip` | Official HF card/downloader recommends this over legacy; contains 2D/3D poses, camera extrinsics/positions and crops | Preferred source to investigate for continuous E0 trajectories; inspect schema, recording coverage, units, confidence and timing before choosing |
| Refined AssemblyHands | Separate toolkit/data release, `v1-1` joint/data/calibration JSONs and skeleton | Candidate higher-quality annotation source; sparse/valid-image coverage and action-split overlap must be established |

The [AssemblyHands toolkit](https://github.com/facebookresearch/assemblyhands-toolkit)
documents right joints 0–20, left 21–41; its loader uses wrists 20/41 and external
`skeleton.txt`. It has per-frame calibration access and joint validity handling.
Do not apply this ordering to legacy arrays without an explicit mapping. Some
ego images without hands are omitted; file-list position is not a video clock.
The toolkit warns that original Assembly101 videos are not necessarily
synchronized or rectified; its own rectified release is a distinct product.

For each candidate source, still establish authoritative full joint names,
hand identity, units, world axes, frame/timestamp origin, validity semantics,
recording coverage and mapping to recognition intervals. The old audit only
inferred millimetres for legacy inputs; it did not prove them. No source is
declared baseline-ready yet. Prefer a verified cleaned sample; if access or
schema prevents this, report the blocker rather than silently use legacy poses.

## Calibration and acquisition sequence

Toolkit pinned revision: `8d4b7f1bce3f99874514199aae7504f78ebf3517`.
Its calibration examples are not a universal rig calibration. Exact matching
recording identity, camera model, dimensions, distortion, units and transform
direction must be checked; ego fisheye and fixed OpenCV models differ.

1. Confirm dataset access; run `scripts/data/inventory_hf.py` to pin HF revision
   and list root plus annotation metadata paths/sizes without downloading assets.
2. Acquire only reviewed small label/vocabulary/split assets at that revision.
   Record hashes and inspect actual memberships and columns.
3. Inspect archive directory metadata for selective cleaned pose/calibration
   access. Require HTTP range support before any remote ZIP extraction; no
   full-archive fallback. The old archive size estimate is not a verified current
   content length. If selective access cannot be bounded, stop for another plan.
4. Intersect official splits with pose and matching-calibration coverage. Inspect
   a tiny annotation sample, then prepare exact recording/view/interval/byte rows.
5. Review the pilot and storage total before video downloads. Clip boundaries
   do not imply a smaller network transfer when the source is a full MP4.

All downloaded data live outside Git under the dedicated project's `data/`.
Only manifests, scripts, compact provenance summaries and diagnostics belong in
Git. Access state, source revision and sample-level validation remain blockers.
