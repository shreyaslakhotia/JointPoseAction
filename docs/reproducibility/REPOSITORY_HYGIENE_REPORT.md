# Final repository hygiene and main consolidation

## Scope and branch history

The user authorized consolidation on 4 November 2026. Windows `main` first
fast-forwarded from `5567cb1` to the completed-result revision
`58265c9336412ef6a0efc2a0dcb5f4dc271b0013`, then incorporated the prepared
`chore/repository-hygiene` tip `eb28d7ae7dc7219c7a379e02dc7e27fac31f2212`
with one final ordinary merge commit. No force-push, reset, rebase or history
rewrite is used. Both development and hygiene histories remain ancestors of
`main`; exact historical training revisions remain available after branch cleanup.
The former hygiene report and map are archived under
`legacy_files/historical_reports/pre_consolidation/hygiene/`.

`main` is the authoritative branch. After successful validation, the final
commit is pushed and local `HEAD` is checked against `git ls-remote origin
refs/heads/main`. Only then are obsolete, fully merged branches removed and the
clean registered hygiene worktree removed using Git. The final chat handoff
records the actual resulting SHA and removal outcome; this committed report
records the procedure without a self-referential commit hash.

## Preserved science and material

- All 710 files in the original hygiene baseline remain recoverable in the
  current tree, verified against the original working/canonical hashes.
- All 711 tracked files from the completed development revision `58265c9`
  (30,005,416 Git-blob bytes) have explicit preserved destinations, including
  the completed B0/B1 comparison and history/provenance record.
- The 76 original protected research/configuration/test/provenance interfaces
  retain their paths. All scientific implementation bytes, preprocessing,
  class mapping, configurations, source pins and selected-checkpoint references
  are unchanged. The preservation helper only resolves moved historical paths;
  the hygiene validator additionally checks the completed development inventory.
- The prepared moves of 570 historical files are reused: old reconstruction
  prototypes, notebooks, segmentation/visualization scripts, mistake annotation
  snapshots, lightweight experiment artifacts and Phase 0 reports.
- Superseded Phase 1C reports, prior overviews and original versions of ten phase
  logs are explicitly archived. Current copies of those logs identify historical
  pending statements and point to completed results.

Staged Git-object checks also confirm that all **710 original baseline blobs**
and all **711 completed-development blobs** are preserved exactly, beyond
working-tree line-ending equivalence.

The complete [migration map](FILE_MIGRATION_MAP.md),
[original inventory](../../reproducibility/manifests/repository_hygiene.json) and
[completed-development inventory](../../reproducibility/manifests/main_consolidation.json)
record destinations and fingerprints. The short [archive inventory](../../legacy_files/README.md)
describes usability limits. Historical hardcoded paths and external dependencies
are preserved as historical context, not claimed newly runnable.

The active recognition packages remain `e0/` and `phase1c/`, alongside `configs/`,
`scripts/`, `tests/`, `docs/` and `reproducibility/`. Retaining package names avoids
an unnecessary packaging migration across tested CLI/checkpoint interfaces.
TCN and HandFormer remain available as future fusion backbones. No new geometry,
reconstruction, training or fusion functionality is added.

## Completed experiment record

The [final comparison](../experiments/DUAL_BACKBONE_COMPARISON.md) records the
same frozen test population: 9,760 segments, 67 recordings and 100 classes.

| Frozen model | Top-1 | Top-5 | Macro-F1 |
|---|---:|---:|---:|
| Original TCN | 28.08% | 54.89% | 9.22% |
| Balanced TCN | 21.53% | 49.26% | 11.44% |
| HandFormer | 36.31% | 65.92% | 17.09% |

All four Phase 1C jobs completed `0:0`: preparation 243530, balanced TCN 243531,
HandFormer 243532 and final evaluation 243533. These are user-supplied cluster
measurements, not a local rerun of training or prediction evaluation. Source
revisions, checkpoint hashes, first/selected/final history snapshots, resources
and uncertainty limits are preserved in
[phase1c_completed_243533.json](../../reproducibility/experiments/phase1c_completed_243533.json).
The dataset SHA-256 remains
`f45c749371568c29f840d2613d9ed20e9859e9020d1555f039882196cd133b59`.

## Validation

The complete CPU suite passes: **91 tests**, including all original E0/Phase 1C
regressions and five historical-path regressions. The pinned official
HandFormer CPU forward/backward test runs against the existing local upstream
checkout; no dependency or dataset download is required. Validation evidence,
command/import checks, shell syntax checks and bounded staged-content review
are recorded in
[main_consolidation_validation.json](../../reproducibility/manifests/main_consolidation_validation.json).
The original dated preservation checks still pass: 20 source snapshots,
540 lightweight artifacts and no failures. Two upstream patch reconstructions
remain explicitly deferred as before; YAML snapshots are preserved by hash,
not claimed semantically validated by the dependency-free checker.

Git checks cover merge conflict markers, whitespace errors, tracked file sizes,
credential patterns and unchanged scientific/configuration interfaces. These
checks establish repository preservation and CPU consistency, not reproduction
of scientific model scores or remote checkpoint availability.

## Local and cluster boundaries

Local ignored `output_visualization.mp4` (2,273,834 bytes), `__pycache__/`, raw
videos, external clones and source reviews remain in place and excluded from
Git. No ignored data is destructively removed or copied into the archive.
The clean registered hygiene worktree contains no unique source after merge;
its removal uses Git's own clean-worktree safety checks.

The cluster checkout, datasets, environments, caches, checkpoints and output
folders under `/projects/jointposeaction` are not accessed, reorganized or
deployed by this task. No GPU job, training, reconstruction or experiment is
submitted. Absolute pose/video synchronization, metric hand/object calibration,
participant overlap and single-seed comparison limits remain scientific
constraints for subsequent geometry/fusion work.
