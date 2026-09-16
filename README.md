# JointPoseAction

A research project on 3D hand and object pose estimation from egocentric assembly videos. The long-term goal is to improve pose tracking in difficult cases such as heavy hand-object occlusion and fast motion blur, supporting fine-grained action recognition and procedural mistake detection.

## Project Overview

This project uses the Assembly101 and AssemblyHands datasets, with the four egocentric HMC camera views as the primary input. The first stage focuses on understanding where baseline annotations and pose estimates fail before adding stronger models and temporal reasoning.

Current Month 1 activities:

- Set up a reproducible Python and Conda environment.
- Download a small subset of baseline videos known to contain annotation errors.
- Build OpenCV-based visualizers for inspecting video frames and annotations.
- Project 3D hand joint coordinates into 2D image frames to compare annotations with the visible hands.
- Record failure cases involving occlusion, motion blur, calibration, and temporal misalignment.

Future experiments will investigate foundation models such as SAM-3D and FoundationPose, together with temporal modeling for more stable tracking across frames.

## Tech Stack

- Python 3.10
- Conda
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook
- Hugging Face Hub

## Local Setup

Create the Conda environment and install the project dependencies:

```bash
conda create -n jointposeaction python=3.10 -y
conda activate jointposeaction
pip install opencv-python numpy matplotlib jupyter huggingface_hub
```

If the Hugging Face repository requires authentication, log in once with:

```bash
huggingface-cli login
```

Download the initial three-video subset:

```bash
python download_subset.py
```

The script writes videos to `data/raw_videos/`. The default Hugging Face repository can be overridden when the dataset mirror uses a different repository ID:

```bash
python download_subset.py --repo-id <owner>/<dataset-repository>
```

## Directory Structure

```text
JointPoseAction/
├── data/
│   ├── raw_videos/       # Downloaded source videos; ignored by Git
│   ├── annotations/      # Local annotation files and calibration data
│   └── processed/        # Extracted frames, projections, and derived data
├── notebooks/            # Exploratory analysis and visualization notebooks
├── src/                  # Reusable dataset, projection, and modeling code
├── download_subset.py    # Download the Month 1 Assembly101 subset
└── README.md
```

Large videos, extracted frames, generated plots, and other local experiment outputs should remain under `data/` and should not be committed to Git. Add a `.gitignore` entry for these paths before downloading a large dataset subset.

## Project Roadmap

### Month 1: Baseline Investigation

- Finalize the local environment and data layout.
- Download and verify the selected Assembly101 videos.
- Understand Assembly101 and AssemblyHands camera, annotation, and calibration formats.
- Implement 3D-to-2D hand-joint projection and OpenCV overlays.
- Catalogue annotation errors and difficult visual conditions.

### Month 2: Baseline Reproduction

- Reproduce a suitable hand and object pose baseline on the four HMC views.
- Establish metrics and evaluation splits.
- Compare baseline predictions with the visualized ground-truth annotations.

### Month 3: Occlusion and Motion Robustness

- Develop occlusion and motion-blur failure categories.
- Evaluate temporal smoothing and sequence-based models.
- Measure performance separately for visible, occluded, and blurred examples.

### Month 4+: Foundation Models and Integration

- Investigate SAM-3D and FoundationPose for hand-object pose refinement.
- Fuse hand and object representations across time.
- Run ablations against the reproduced baseline.
- Document limitations, qualitative results, and directions for the final system.

## Data and Citation

Please follow the official Assembly101 and AssemblyHands terms of use, download procedures, and citation requirements. Do not commit dataset files or credentials to this repository.
