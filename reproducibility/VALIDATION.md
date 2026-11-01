# Preservation checks — 1 November 2026

These are lightweight preservation checks, not ML correctness or a cluster
runtime validation. No source implementation, dataset or environment was edited.

- Parsed all ten custom Python files and all six code cells in the existing
  notebook using AST, without importing model code or generating bytecode.
- Parsed all fifteen saved YAML environment/model/reference files and the
  manifest/audit JSON files. Migration document relative links resolve.
- Reconstructed both changed SAM3D vendor source files in a temporary directory
  from the exact upstream Git blobs plus `local-modifications.patch`; both match
  current workstation bytes. `git apply --check` succeeded on that clean base.
- Verified custom_inference.py and all nine checkpoint YAML snapshots against
  their originals. Source/config SHA256 mapping is in preservation_checksums.json.
- Recorded SHA256/size for all 540 retained original small artifacts. Dataset
  files/model weights were not hashed or copied; their sizes and known download
  provenance are recorded separately.
- Credential-pattern scan of candidate text files found no GitHub/HF token,
  AWS access-key, private-key block or password-bearing URL patterns. No secret
  values were printed; this is a bounded pattern review, not a guarantee about
  undisclosed external files.
- `.gitignore` checks exclude raw data, the installer, nested clones/weights and
  generated RGB/RGBA packages while permitting the custom snapshots, small
  evidence images, original CSVs and report JSONs. No new candidate exceeds
  5 MiB (largest retained original image is 2,322,454 bytes).
- Before commit, the remote main branch matched initial local HEAD
  ac80eb734752677657a2eee203edf83ec1e8ffdc. Commit/push verification is reported in
  the final conversation; this file does not embed its own circular commit SHA.

Intentional limits: exported environments were not installed/solved on another
machine; CUDA/inference/training, redownloads, new extraction code and cluster
access were not attempted. Original source/code defects remain as documented in
the Phase 0 reports. The two ignored vendor modifications and custom file stay
in their original working locations as well as being recoverable from Git.
