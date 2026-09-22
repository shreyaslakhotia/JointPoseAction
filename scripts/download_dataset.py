import os
import glob
import zipfile
import fsspec
from pathlib import Path
from huggingface_hub import snapshot_download

def download_calibrations_remotely():
    print("\n--- Phase 1: Extracting Calibrations from 72GB ZIP Remotely ---")
    zip_url = "https://huggingface.co/datasets/cvml-nus/assembly101/blob/main/AssemblyPoses.zip"
    
    # fsspec allows us to read a ZIP file over the internet without downloading it
    print("Establishing remote HTTP connection to AssemblyPoses.zip...")
    fs = fsspec.filesystem('http')
    
    try:
        with fs.open(zip_url, 'rb') as remote_file:
            with zipfile.ZipFile(remote_file) as z:
                # Find the calibration files inside the zip
                all_files = z.namelist()
                calib_files = [f for f in all_files if 'extrinsics' in f.lower() or 'intrinsics' in f.lower() or 'calib' in f.lower()]
                
                if not calib_files:
                    print("Could not find calibration files in the ZIP root.")
                    return
                
                print(f"Found {len(calib_files)} calibration files. Streaming them locally...")
                os.makedirs("data/calibrations", exist_ok=True)
                
                for calib_file in calib_files:
                    # Extract only these tiny files directly to our local folder
                    z.extract(calib_file, "data/calibrations")
                    print(f" -> Saved {calib_file}")
    except Exception as e:
        print(f"Remote extraction failed: {e}")
        print("Ensure you have a stable internet connection for the HTTP stream.")

def download_video_batch(num_videos=50):
    print(f"\n--- Phase 2: Downloading {num_videos} Exocentric Videos ---")
    csv_dir = Path("mistake_repo/annots")
    
    if not csv_dir.exists():
        raise FileNotFoundError("Could not find mistake_repo/annots/. Did you run the sparse checkout commands?")

    # Extract unique video IDs from the CSV filenames
    csv_files = sorted(glob.glob(str(csv_dir / "*.csv")))
    video_ids = [os.path.basename(f).replace('.csv', '') for f in csv_files][:num_videos]
    
    for i, vid in enumerate(video_ids, 1):
        print(f"\n[{i}/{num_videos}] Processing {vid}...")
        
        # Download Exocentric MP4s (C*.mp4)
        snapshot_download(
            repo_id="cvml-nus/assembly101",
            repo_type="dataset",
            allow_patterns=f"recordings/{vid}/C*.mp4",
            local_dir="data/raw_videos"
        )
        
        # Download 60fps Pose JSON
        snapshot_download(
            repo_id="cvml-nus/assembly101",
            repo_type="dataset",
            allow_patterns=f"poses@60fps/{vid}.json",
            local_dir="data/poses"
        )

if __name__ == "__main__":
    # 1. Grab the camera parameters remotely first
    download_calibrations_remotely()
    
    # 2. Download the 50 videos and their JSONs
    download_video_batch(num_videos=50)
    
    print("\n✅ Pipeline complete! You are ready for (Multi-GPU Processing).")
