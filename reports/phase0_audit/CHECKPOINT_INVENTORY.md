# Checkpoint filesystem inventory

1 November 2026. Sizes only; no checkpoint load or full binary checksum. Relative paths under SAM 3D repository. Zero-byte files flagged below are not referenced by current `pipeline.yaml`.

| Path | Bytes |
|---|---:|
| `checkpoints/hf/pipeline.yaml` | 3,548 |
| `checkpoints/hf/slat_decoder_gs.ckpt` | 171,476,155 |
| `checkpoints/hf/slat_decoder_gs.yaml` | 576 |
| `checkpoints/hf/slat_decoder_gs_4.ckpt` | 170,269,801 |
| `checkpoints/hf/slat_decoder_gs_4.yaml` | 575 |
| `checkpoints/hf/slat_decoder_mesh.ckpt` | 363,726,862 |
| `checkpoints/hf/slat_decoder_mesh.pt` | 363,728,714 |
| `checkpoints/hf/slat_decoder_mesh.yaml` | 300 |
| `checkpoints/hf/slat_encoder.ckpt` | 173,263,986 |
| `checkpoints/hf/slat_encoder.yaml` | 268 |
| `checkpoints/hf/slat_generator.ckpt` | 4,906,537,684 |
| `checkpoints/hf/slat_generator.yaml` | 1,986 |
| `checkpoints/hf/ss_decoder.ckpt` | 147,609,242 |
| `checkpoints/hf/ss_decoder.yaml` | 244 |
| `checkpoints/hf/ss_encoder.ckpt` | 119,085,402 |
| `checkpoints/hf/ss_encoder.safetensors` | 0 |
| `checkpoints/hf/ss_encoder.yaml` | 231 |
| `checkpoints/hf/ss_generator.ckpt` | 6,690,136,964 |
| `checkpoints/hf/ss_generator.yaml` | 5,076 |
| `checkpoints/sam3d_objects_v1.pth` | 0 |

Adjacent SAM2 large checkpoint: `/home/mlda/shreyas_projects/sam2/checkpoints/sam2.1_hiera_large.pt` — 898,083,611 bytes.

External DINO checkpoint — 1,217,607,321 bytes; cached MoGe model — 1,256,823,446 bytes. See environment report for exact paths/snapshot. Presence and size do not establish numerical integrity or that a checkpoint was used successfully.
