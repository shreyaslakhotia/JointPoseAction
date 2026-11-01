# Preserved small mistake-annotation snapshot

`annots/` preserves all 328 original CSV files found on gpu2 (182,677 bytes),
without changing their bytes or pretending this directory is a Git clone.
The exact source revision is unknown.

Official source and schema:
https://github.com/assembly-101/assembly101-mistake-detection

Research references: Assembly101 (Sener et al., CVPR 2022) and Every Mistake
Counts in Assembly (Ding et al., 2023). The official repository provides the
dataset license/terms (CC BY-NC 4.0) and interpretation of mistake/correction
remarks. This snapshot is data, not custom annotation ground truth or a
reimplementation of the upstream benchmark.

The 20 downloaded recording IDs and 87 legacy pose filenames are separately
listed under `reproducibility/datasets/`. The local CSV time base is not
independently verified; see the Phase 0 audit before pairing intervals to raw
60 fps videos. Checksums are in `reproducibility/small_artifacts.json`.
