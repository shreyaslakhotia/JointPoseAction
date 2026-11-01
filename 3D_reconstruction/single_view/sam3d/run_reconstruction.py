import os
import json
import subprocess

# --- Configuration Paths ---
PROJECT_ROOT = "/home/mlda/shreyas_projects/JointPoseAction"
DATA_DIR = os.path.join(PROJECT_ROOT, "experiments/reconstruction_data/session_01")
META_PATH = os.path.join(DATA_DIR, "metadata.json")

# Pointing to the official repo
SAM3D_REPO_DIR = os.path.join(PROJECT_ROOT, "3D_reconstruction/single_view/sam3d_repo")
SAM3D_DEMO_SCRIPT = "demo.py" # We just call the script name now since we change directories
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "3D_reconstruction/single_view/sam3d/output")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading metadata from {META_PATH}...")
    with open(META_PATH, 'r') as f:
        metadata = json.load(f)

    ranking = metadata.get("ranking_by_mask_area", [])
    if not ranking:
        raise ValueError("No camera ranking found in metadata.json!")
    
    best_cam = ranking[0]
    cam_data = metadata["cameras"][best_cam]
    
    print(f"🏆 Best View Selected: {best_cam} ({cam_data['foreground_pixels']:,} foreground pixels)")

    rgb_path = os.path.join(DATA_DIR, cam_data["rgb_file"])
    mask_path = os.path.join(DATA_DIR, cam_data["mask_file"])

    command = [
        "python", SAM3D_DEMO_SCRIPT,
        "--image_path", rgb_path,
        "--mask_path", mask_path,
        "--output_dir", OUTPUT_DIR,
        "--export_mesh",          
        "--export_pointcloud",    
        "--no_gaussian"           
    ]

    print("\n🚀 Executing SAM 3D Objects Inference:")
    print(" ".join(command) + "\n")

    # Inject Memory Management Environment Variable to prevent OOM
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    try:
        # Notice we added cwd=SAM3D_REPO_DIR here!
        subprocess.run(command, env=env, cwd=SAM3D_REPO_DIR, check=True)
        print(f"\n🎉 Reconstruction Complete! Check the outputs in:\n{OUTPUT_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Reconstruction failed. Error code: {e.returncode}")

if __name__ == "__main__":
    main()
