import json
import cv2
import numpy as np
from pathlib import Path

# --- Configuration ---
VIDEO_ID = "nusar-2021_action_both_9011-b06b_9011_user_id_2021-02-01_154253"
CAMERA_ID = "C10095"  
VIDEO_PATH = f"data/raw_videos/recordings/{VIDEO_ID}/{CAMERA_ID}_rgb.mp4"
POSE_PATH = f"data/poses/poses@60fps/{VIDEO_ID}.json"

# We use the toolkit's calibration file which has the perfect ModelView matrices!
CALIB_PATH = "assemblyhands-toolkit/calib/nimble_json_calib/nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345.json"
OUTPUT_PATH = "output_custom_vis.mp4"

def load_camera_matrices():
    with open(CALIB_PATH, 'r') as f:
        calib_list = json.load(f)
        
    cam_data = None
    for item in calib_list:
        if item.get('Camera', {}).get('SerialNo') == f"{CAMERA_ID}_rgb":
            cam_data = item['Camera']
            break
            
    if not cam_data:
        raise ValueError(f"Could not find {CAMERA_ID} in the calibration JSON!")

    # 1. Exact Intrinsics
    K = np.array([
        [cam_data['fx'], 0, cam_data['cx']],
        [0, cam_data['fy'], cam_data['cy']],
        [0, 0, 1.0]
    ], dtype=np.float32)
    
    # 2. Exact Lens Distortion (k1, k2, p1, p2, k3)
    dist_coeffs = np.array([
        cam_data['k1'], cam_data['k2'], 
        cam_data['p1'], cam_data['p2'], 
        cam_data['k3']
    ], dtype=np.float32)
    
    # 3. Exact Extrinsics (ModelViewMatrix is already World-to-Camera!)
    # No manual inversion needed.
    E = np.array(cam_data['ModelViewMatrix'], dtype=np.float32)
    R = E[:3, :3]
    T = E[:3, 3].reshape(3, 1)
    
    rvec, _ = cv2.Rodrigues(R)
    
    return K, dist_coeffs, rvec, T

def draw_skeleton(frame, points_2d, color=(0, 255, 0)):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),# Ring
        (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
    ]
    
    # Safety Check: If math exploded and produced NaNs, skip drawing
    if np.isnan(points_2d).any():
        return
        
    points_2d = points_2d.astype(int)
    h, w = frame.shape[:2]
    
    # Draw connections
    for pt1, pt2 in edges:
        if pt1 < len(points_2d) and pt2 < len(points_2d):
            # Safety Check: Don't draw lines to infinity if points project wildly off-screen
            if (0 <= points_2d[pt1][0] <= w*2) and (0 <= points_2d[pt1][1] <= h*2):
                cv2.line(frame, tuple(points_2d[pt1]), tuple(points_2d[pt2]), color, 2)
            
    # Draw joints
    for pt in points_2d:
        if (0 <= pt[0] <= w*2) and (0 <= pt[1] <= h*2):
            cv2.circle(frame, tuple(pt), 4, (0, 0, 255), -1)

def main():
    print("Loading perfect camera matrices...")
    K, dist_coeffs, rvec, tvec = load_camera_matrices()

    print("Loading pose data...")
    with open(POSE_PATH, 'r') as f:
        pose_data = json.load(f)
    
    poses_by_frame = {frame['frame_index']: frame['landmarks'] for frame in pose_data}

    print(f"Processing video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    MAX_FRAMES = 500  
    frame_count = 0
    
    while cap.isOpened() and frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
            
        landmarks = poses_by_frame.get(frame_count, {})
        
        # '0' is Left Hand, '1' is Right Hand
        for hand_id, color in [('0', (255, 150, 0)), ('1', (0, 255, 255))]:
            if hand_id in landmarks:
                points_3d = np.array(landmarks[hand_id], dtype=np.float32)
                
                # Project!
                points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)
                points_2d = points_2d.reshape(-1, 2)
                
                draw_skeleton(frame, points_2d, color=color)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{MAX_FRAMES} frames...")

    cap.release()
    out.release()
    print(f"✅ Success! Saved aligned video to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
