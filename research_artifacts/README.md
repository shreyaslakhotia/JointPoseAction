# Research artifacts

This directory contains lightweight, attributable review figures and measured
metadata. Large datasets, native masks, model checkpoints and full-resolution
outputs stay in the dedicated 50 GB project allocation.

- [Segmentation pilot](segmentation_pilot/PILOT_SELECTION.md): three diagnostic
  candidates, source-frame figures, calibration evidence and the SAM 3 workflow.
  **No reconstruction-ready pilot has yet been accepted.**
- [Reconstruction preparation](reconstruction_benchmark/READINESS.md): CPU
  implementation and synthetic validation only. Real experiments remain blocked.

Complete segmentation files:
`/projects/jointposeaction/experiments/segmentation_pilot/`
(real path `/projects/_ssd/jointposeaction/experiments/segmentation_pilot/`).
Consult each manifest for immutable source revisions, file hashes and job IDs.
Figures label raw stream PTS rather than claiming synchronized exposures.

Assembly101 RGB figures are research derivatives of
[Assembly101](https://huggingface.co/datasets/cvml-nus/assembly101), by Sener et al.,
CVPR 2022, under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Changes are downsampling/cropping, external labels and explicitly identified
prediction overlays. Native originals are retained. Calibration derives from the
[AssemblyHands toolkit](https://github.com/facebookresearch/assemblyhands-toolkit),
Ohkawa et al., CVPR 2023. Model outputs do not create independent ground truth or
imply endorsement by either dataset's authors.
