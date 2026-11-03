# Repository hygiene report

## Isolation and experiment safety

Original production commit: `fcb0a40db35c1c20da76df5fa90fa7a9179d63ad` on
`development/e0-baseline`. All changes are made on `chore/repository-hygiene` in
the isolated local worktree `data/repository_hygiene_worktree`. The parent
production worktree and cluster checkout are not edited, switched or deployed.
Final reorganization commit: `6258c647420fc157c301ad96505251683eab9fb7`.
The report/evidence finalization is a documentation-only successor; the exact
review head is `git rev-parse chore/repository-hygiene`. This branch is not merged
into production.

The latest user-supplied accounting reports prepare 243530 completed `0:0` in
31 seconds; B0-balanced 243531 completed `0:0` in 17m19s; B1-HandFormer 243532
and evaluation 243533 pending. No hygiene action queries, cancels, resubmits,
alters or interrupts these jobs. No new held-out recognition result is claimed.

The queued B1/evaluation jobs enter `phase1c.pipeline` from
`/projects/jointposeaction/repo` and require production commit `fcb0a40`.
Their closure includes `phase1c/`, shared `e0/` modules, the two recognition
configurations, `reproducibility/upstreams/handformer.json`, and the cluster
environment/stage wrappers. Every one retains its production path and bytes in
this branch except the unrelated preservation checker, whose changes resolve
archival paths only. Scientific source and settings are not refactored.

## Classification and preservation

The baseline contains 710 tracked files totaling 29,960,681 Git-blob bytes.
The complete inventory and old/new paths, classifications, rationale and
fingerprints are in
[repository_hygiene.json](../../reproducibility/manifests/repository_hygiene.json).

- 137 baseline files retain their active paths and content.
- 570 historical files move to `legacy_files/` using `git mv`.
- Three original versions are preserved before replacing the root README,
  updating ignore policy, or adding path resolution to the preservation checker.
- All 710 baseline contents remain recoverable in the current tree, in addition
  to Git history. The original README, ignore policy and preservation helper
  receive small explicit snapshots because their replacements serve different
  current purposes. No large duplicate dataset or artifact copy is created.

The archive separates historical reconstruction source, notebooks, old
experiments and mistake annotation CSV snapshots, deprecated exploratory
utilities, diagnostic tooling and dated Phase 0 reports. Filenames and relevant
directory relationships are preserved. Archived source and reports are not
rewritten. Historical hardcoded workstation paths and original working-directory
assumptions mean these are recoverable snapshots, not newly validated applications.

No original FYP proposal, PDF, Word or presentation document is tracked in the
baseline. None was removed or recreated. The existing research scope and verified
experimental documents are preserved.

## Retained active structure

`e0/` and `phase1c/` remain separate recognition packages, sharing their existing
data/runtime/evaluation helpers. `configs/`, `scripts/cluster/`, `scripts/data/`,
`tests/`, `docs/` and `reproducibility/` remain their canonical interfaces.
No empty `src/` hierarchy, packaging migration, numerical refactor or model
behavior change is introduced. Detailed paths are in the
[file migration map](FILE_MIGRATION_MAP.md).

The dated restoration/migration guides move unchanged to the archive; short
current wrappers preserve their entry paths and explain the original-layout
context. Root README and documentation indexes now distinguish current
recognition work, future geometry/fusion gates, historical records and external
cluster artifacts. No research functionality is added.

## Manifest paths and generated artifacts

All original reproducibility manifests, upstream revisions, local vendor
patches, environment snapshots, selection metadata and experimental evidence
remain byte-identical. Their old paths continue to describe the dated snapshots.
The preservation checker resolves those historical file paths through the new
map, while checking current tracked syntax at its actual path. Committed-content
verification is retained for line-ending-only differences.

Ignore coverage is mirrored at moved archive paths, preserving the selected
historical overlays and annotation CSV exceptions. Active runtime experiment
directories, environments, caches and external clones are also excluded. Raw
videos, full pose archives, checkpoints, generated feature arrays and Conda
environments stay outside Git. Existing tracked lightweight images are retained;
the largest baseline file is 2,322,454 bytes.

The cluster's `$JPA_ROOT/experiments/e0` and `experiments/phase1c` are external to
the source checkout. Moving the old tracked `experiments/` directory does not
move those runs. Frozen dataset SHA-256 remains
`f45c749371568c29f840d2613d9ed20e9859e9020d1555f039882196cd133b59`.
Checkpoint interfaces/configurations remain unchanged. Actual remote B0
checkpoint bytes and dataset availability are not independently reverified by
this local task; their references and recorded provenance are preserved.

## Validation

Before moves, all 86 original CPU tests passed in the isolated worktree, using
the existing local environment and an existing pinned HandFormer checkout via
`JPA_HANDFORMER_ROOT`; no upstream dependency was downloaded or installed.
After the moves, all **91 CPU tests** passed (the same original 86 plus five
new path-resolution regressions). The existing pinned HandFormer model's real
CPU forward/backward check ran without a skip. Eight recognition/data CLI help
entry points and all 25 cluster shell/Slurm files passed lightweight checks.

The complete inventory check found **710/710 exact original working-byte matches**,
zero missing destinations, unchanged paths/content for 76 protected research and
provenance files, no broken links across 29 current documentation files, and ten
passing ignore-policy probes. The original dated manifests validate through the
new resolver: 20 preserved source snapshots checked with line-ending-only Git
verification; two patch reconstructions remain explicitly deferred as before;
all 540 small artifacts pass (326 exact, 214 line-ending-only). Python, notebook
and JSON syntax checks pass. YAML snapshots are hash-preserved; the dependency-free
checker does not claim YAML semantic validation.

Independent post-move review confirmed original Git blobs as well as working
bytes. A bounded final scan of 511 text files found zero checked credential-pattern
matches; no candidate file exceeds 5 MiB. The largest new metadata is the
483,043-byte complete migration map. Post-commit verification passed at structural commit `6258c647420fc157c301ad96505251683eab9fb7`.
Machine-readable evidence is in
[repository_hygiene_validation.json](../../reproducibility/manifests/repository_hygiene_validation.json).
The final production worktree check remained clean on `development/e0-baseline`
at `fcb0a40db35c1c20da76df5fa90fa7a9179d63ad`.

A bounded credential-pattern scan of 498 baseline tracked text files found no
matches for the checked token/key/private-key/password-URL patterns. It did not
inspect ignored token caches or print candidate secret values. This is a bounded
check, not a general guarantee about arbitrary historical content.

## Review and remaining limitations

Merge/deployment awaits user approval and completion and safe preservation of the active
experiment artifacts. The branch must not replace the queued jobs' checkout.
Historical restoration commands require their original revision/layout or
deliberate path adaptation. Absolute video/pose synchronization, metric
hand/object calibration, recognition performance and future fusion design are
unchanged research limitations. A passing hygiene check does not establish any
new scientific result.
