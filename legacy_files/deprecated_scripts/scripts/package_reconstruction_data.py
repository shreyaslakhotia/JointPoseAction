import os
import json
import glob
import shutil
import cv2
import numpy as np

# --- Path Configuration ---
PROJECT_ROOT = "/home/mlda/shreyas_projects/JointPoseAction"
FRAMES_DIR = os.path.join(PROJECT_ROOT, "experiments/sam2_test/frames")
MASKS_DIR = os.path.join(PROJECT_ROOT, "experiments/sam2_test/sam3_binary_masks")
CALIB_PATH = os.path.join(
    PROJECT_ROOT, 
    "assemblyhands-toolkit/calib/nimble_json_calib/nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345.json"
)

OUTPUT_BASE = os.path.join(PROJECT_ROOT, "experiments/reconstruction_data/session_01")
OUT_RGB = os.path.join(OUTPUT_BASE, "rgb")
OUT_MASKS = os.path.join(OUTPUT_BASE, "masks")
OUT_RGBA = os.path.join(OUTPUT_BASE, "rgba")

CAMERAS = ["C10095", "C10115", "C10118", "C10119", "C10379", "C10390", "C10395", "C10404"]

def load_calibration():
    if not os.path.exists(CALIB_PATH):
        raise FileNotFoundError(f"Calibration JSON not found at: {CALIB_PATH}")
    with open(CALIB_PATH, 'r') as f:
        data = json.load(f)
    calib_by_cam = {}
    for item in data:
        cam_info = item.get('Camera', {})
        serial = cam_info.get('SerialNo', '')
        cam_id = serial.split('_')[0]
        if cam_id in CAMERAS:
            calib_by_cam[cam_id] = cam_info
    return calib_by_cam

def main():
    os.makedirs(OUT_RGB, exist_ok=True)
    os.makedirs(OUT_MASKS, exist_ok=True)
    os.makedirs(OUT_RGBA, exist_ok=True)

    print("Loading camera calibrations...")
    calib_data = load_calibration()

    metadata = {
        "session_id": "session_01",
        "description": "Paired RGB, binary masks, RGBA cutouts, and camera calibrations for 3D reconstruction",
        "cameras": {},
        "ranking_by_mask_area": []
    }

    views_summary = []

    print("\nPackaging viewpoints...")
    for cam_id in CAMERAS:
        # 1. Locate binary mask
        mask_candidates = glob.glob(os.path.join(MASKS_DIR, f"{cam_id}*_mask.png"))
        if not mask_candidates:
            print(f"⚠️ Warning: No binary mask found for {cam_id} in {MASKS_DIR}")
            continue
        src_mask_path = mask_candidates[0]

        # 2. Locate clean frame (pick a stable frame in the static window, e.g. frame 0005 or first available)
        frame_candidates = sorted(glob.glob(os.path.join(FRAMES_DIR, f"{cam_id}_*.jpg")))
        if not frame_candidates:
            print(f"⚠️ Warning: No raw frame found for {cam_id} in {FRAMES_DIR}")
            continue
        
        # Pick frame 0005 if present, otherwise default to first available
        target_frame = frame_candidates[min(4, len(frame_candidates) - 1)]
        src_rgb_path = target_frame

        # 3. Read image and mask for validation and RGBA generation
        img_bgr = cv2.imread(src_rgb_path)
        mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)

        if img_bgr is None or mask is None:
            print(f"❌ Failed to load image or mask for {cam_id}")
            continue

        h, w = img_bgr.shape[:2]

        # 4. Strictly COPY files to target directories (preserving originals)
        dst_rgb_name = f"{cam_id}_rgb.png"
        dst_mask_name = f"{cam_id}_mask.png"
        dst_rgba_name = f"{cam_id}_rgba.png"

        dst_rgb_path = os.path.join(OUT_RGB, dst_rgb_name)
        dst_mask_path = os.path.join(OUT_MASKS, dst_mask_name)
        dst_rgba_path = os.path.join(OUT_RGBA, dst_rgba_name)

        # Save RGB as lossless PNG
        cv2.imwrite(dst_rgb_path, img_bgr)
        # Copy binary mask
        shutil.copy2(src_mask_path, dst_mask_path)

        # 5. Generate transparent RGBA cutout
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgba = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = mask
        cv2.imwrite(dst_rgba_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        # 6. Extract calibration matrices
        cam_calib = calib_data.get(cam_id, {})
        fx = float(cam_calib.get('fx', 0.0))
        fy = float(cam_calib.get('fy', 0.0))
        cx = float(cam_calib.get('cx', 0.0))
        cy = float(cam_calib.get('cy', 0.0))
        K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        
        dist = [
            float(cam_calib.get('k1', 0.0)),
            float(cam_calib.get('k2', 0.0)),
            float(cam_calib.get('p1', 0.0)),
            float(cam_calib.get('p2', 0.0)),
            float(cam_calib.get('k3', 0.0))
        ]

        E = np.array(cam_calib.get('ModelViewMatrix', np.eye(4)), dtype=np.float32)
        R = E[:3, :3]
        t = E[:3, 3]

        # Camera center in world coordinates: c = -R^T * t
        camera_center_world = (-R.T @ t).tolist()

        # 7. Quality metrics: Bounding box and foreground area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            bbox = [int(x), int(y), int(x + bw), int(y + bh)]
        else:
            bbox = [0, 0, 0, 0]

        fg_pixels = int(np.count_nonzero(mask == 255))

        metadata["cameras"][cam_id] = {
            "source_frame": os.path.basename(src_rgb_path),
            "rgb_file": f"rgb/{dst_rgb_name}",
            "mask_file": f"masks/{dst_mask_name}",
            "rgba_file": f"rgba/{dst_rgba_name}",
            "image_size": [w, h],
            "foreground_pixels": fg_pixels,
            "bbox_xyxy": bbox,
            "intrinsics_K": K,
            "distortion_coeffs": dist,
            "extrinsics_R": R.tolist(),
            "extrinsics_t": t.tolist(),
            "camera_center_world": camera_center_world
        }

        views_summary.append((cam_id, fg_pixels, bbox))
        print(f"✅ Packaged {cam_id} (Foreground: {fg_pixels:,} px | Box: {bbox})")

    # Sort views by mask area descending
    views_summary.sort(key=lambda item: item[1], reverse=True)
    metadata["ranking_by_mask_area"] = [item[0] for item in views_summary]

    meta_out_path = os.path.join(OUTPUT_BASE, "metadata.json")
    with open(meta_out_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\n🎉 Packaging complete! All files saved to:\n  {OUTPUT_BASE}")
    print(f"📄 Metadata written to: {meta_out_path}")
    print("\nViewpoint Ranking for Baseline A (largest mask coverage to smallest):")
    for rank, (cam, area, bbox) in enumerate(views_summary, 1):
        print(f"  Rank {rank}: {cam} ({area:,} pixels)")

if __name__ == "__main__":
    main()
