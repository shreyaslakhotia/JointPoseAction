# Inspection record and lightweight reproduction recipe

Audit date: 1 November 2026, Asia/Singapore. Reports/diagnostic JSONs only were written. Source/model/data files were read, not edited. Existing report files did not exist at the target directory before creation.

## Scope and boundaries

- Root and adjacent SAM 2 repo inspected; no applicable AGENTS.md found in the searched project locations/ancestor candidates. No subagents were used.
- Root/nested Git state and diffs inspected, credential-bearing URL syntax redacted before display. No Git branch/index/history mutation requested; no commits, fetches or submodule operations performed.
- Selective directory listings only; no recursive video/checkpoint reads, Conda cache scan, large disk-usage crawl, shell history, SSH/AWS credentials or account-secret inspection.
- Header probing all 160 videos; six videos decoded at three selected frame positions. Three pose JSONs parsed sequentially, numerical diagnostics on uniform samples. Small calibrations/images inspected. No GPU model allocation/inference or compiled extension import sweep.
- Diagnostic Python invocations used `PYTHONDONTWRITEBYTECODE=1` and `-B`; numerical work used `OPENBLAS_NUM_THREADS=1`. Ordinary filesystem reads can update atime under the host mount policy; this audit did not change mount flags or guarantee O_NOATIME on library-mediated image/video reads.

## Evidence capture commands

Git: `git --no-optional-locks status --short`, `git status --porcelain=v1 --ignored --untracked-files=normal`, `git ls-files`, `git rev-parse HEAD`, `git branch --show-current`, `git log -6 --oneline`, `git submodule status`, `git ls-files -u`; corresponding revision/status/remote checks in vendor/adjacent repositories. SAM3D `git diff` identifies debug-only changes in inference pipeline and sparse decoder. No `.gitmodules` or unresolved index entries were found. `mistake_repo` is data-only, so `git -C` there reports its parent's revision.

Hardware: `uname -a`, `lscpu`, `free -h`, `df -h . /tmp`, `lspci`, `/etc/os-release`, `/proc/driver/nvidia/version`, NVIDIA GPU directory names, `nvidia-smi`, environment-local `nvcc --version`. No administrative operation or external host connection was used.

Software: inspect `conda-meta/*.json` for selected package names and `lib/python*/site-packages/<package>-*.dist-info` names in each named environment. Short CPU probes print torch version, CUDA build, availability/device count; fyp-pose probe also prints cv2 and NumPy runtime versions. This is not an extension ABI/inference test.

## Reproduction of diagnostic artifacts

The following are specifications of the read-only diagnostic procedure, not new pipeline implementations. Re-run only the bounded sample described, writing new results to another audit location if comparison is needed.

1. **dataset_inventory.json:** enumerate immediate recording folders and their eight MP4s. For each call `ffprobe -v error -show_entries stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,start_time -of json <path>`. Record file size, stderr and output. Match immediate pose/CSV filename by session ID. Parse each of twenty small mistake CSVs with Python csv; count labels/verbs/part names, maximum endpoint, and end<=start rows. Read the literal SPLIT2SEQ assignment from toolkit `src/dataset/__init__.py` using AST literal evaluation, without importing its modules.
2. **pose_diagnostics.json:** full-load only JSON filenames matching `*9011-b06b*`, `*9012-c07c*`, `*9013-c03b*`, sequentially. Count indices, deltas, row hand keys; sample rows with `rows[::max(1,len(rows)//200)]`. For hand arrays count zero xyz and nonfinite coordinates and confidence-zero hands. For c07c's eight fixed cameras, use direct K,D,R,t in OpenCV `projectPoints`; record positive-depth and inside-image fractions. This is a plausibility diagnostic, not measured reprojection accuracy.
3. **calibration_diagnostics.json:** read all twenty small Nimble JSONs; for each fixed camera compare complete camera dictionaries across files and max absolute ModelViewMatrix/fx difference from c07c. Check `det(R)` and `max(abs(R.T@R-I))` for the reference rotations. All twenty include the fixed-camera serials.
4. **mask_diagnostics.json:** read all eight package masks in grayscale. Use `np.unique`, pixel count and OpenCV connectedComponentsWithStats on `(mask>0).astype(uint8)`. Record largest component areas, metadata bbox/source filename, and pixel equality between package RGB and named source JPEG.
5. **extraction_diagnostics.json:** AST-extract only `extract_binary_mask` from `scripts/extract_sam3_masks.py`; execute its definition with NumPy/OpenCV globals, without invoking main. For each of eight input PNGs call with `visualize=False`, compare returned array to stored mask. Read corresponding deprecated SAM2 `*_0005_mask.png` and count foreground/components. No image is written by this function call.
6. **package_diagnostics.json:** compare every packaged mask to its contour-mask source, RGBA alpha to mask, and decoded RGBA BGR channels to packaged RGB using OpenCV. All 24 comparisons pass.
7. **decode_sample.json:** sessions b06b/c07c/c03b, cameras C10095/C10118; open each with cv2.VideoCapture and seek to indices 0, n//2, n-1; read one image each, record success and shape; release. Eighteen successful samples, no saved frames.
8. **Adapter type diagnostic:** AST-extract only `Inference.merge_mask_to_rgba` from `sam3d_repo/notebook/inference.py` with NumPy in globals. Call using tiny 2×2 PIL RGB/L images: AttributeError on mask.astype. Call with a 1×2 RGB ndarray and uint8 `[0,255]` mask: output alpha `[0,1]`. No heavy module imports or model invocation. Source lines 94–99 establish required Boolean/array behavior.

Representative images actually viewed: contour-derived C10095/C10118/C10395/C10404 overlays; original C10118 claimed SAM3 overlay; C10118 packaged clean RGB; deprecated SAM2 C10118_0005 overlay. No GT IoU was computed.

## Repositories at inspection

| Root | Branch | HEAD | Existing modifications |
|---|---|---|---|
| Project | main | ac80eb734752677657a2eee203edf83ec1e8ffdc | No tracked changes; untracked reconstruction/preprocessing/artifact groups listed in main report |
| AssemblyHands | main | 8d4b7f1bce3f99874514199aae7504f78ebf3517 | Clean |
| SAM 3D Objects | main | f91db411c50efee93d8db7aeb323885650f6f722 | Two debug-print modifications, untracked custom_inference.py |
| SAM 2 adjacent | main | 2b90b9f5ceec907a1c18123530e92e794ad901a4 | Clean |

Official web sources were used for model/data interface questions, not to substitute for local execution evidence. Full source URLs are in the main report. The complete historical handover and OOM traceback were not available. No implementation followed the audit.
