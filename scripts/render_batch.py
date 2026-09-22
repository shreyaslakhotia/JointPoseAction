import json
import cv2
import numpy as np
import os
import glob
from pathlib import Path

# --- Configuration ---
CAMERA_ID = "C10095"
OUTPUT_DIR = "output_full_videos"
# The universal calibration file that works for all C10095 views
CALIB_PATH = "assemblyhands-toolkit/calib/nimble_json_calib/nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345.json"

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

    K = np.array([
        [cam_data['fx'], 0, cam_data['cx']],
        [0, cam_data['fy'], cam_data['cy']],
        [0, 0, 1.0]
    ], dtype=np.float32)
    
    dist_coeffs = np.array([
        cam_data['k1'], cam_data['k2'], 
        cam_data['p1'], cam_data['p2'], 
        cam_data['k3']
    ], dtype=np.float32)
    
    E = np.array(cam_data['ModelViewMatrix'], dtype=np.float32)
    R = E[:3, :3]
    T = E[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    
    return K, dist_coeffs, rvec, T

def draw_skeleton(frame, points_2d, color=(0, 255, 0)):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),       
        (0, 5), (5, 6), (6, 7), (7, 8),       
        (0, 9), (9, 10), (10, 11), (11, 12),  
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20) 
    ]
    if np.isnan(points_2d).any():
        return
        
    points_2d = points_2d.astype(int)
    h, w = frame.shape[:2]
    
    for pt1, pt2 in edges:
        if pt1 < len(points_2d) and pt2 < len(points_2d):
            if (0 <= points_2d[pt1][0] <= w*2) and (0 <= points_2d[pt1][1] <= h*2):
                cv2.line(frame, tuple(points_2d[pt1]), tuple(points_2d[pt2]), color, 2)
            
    for pt in points_2d:
        if (0 <= pt[0] <= w*2) and (0 <= pt[1] <= h*2):
            cv2.circle(frame, tuple(pt), 4, (0, 0, 255), -1)

def process_video(video_id, K, dist_coeffs, rvec, tvec):
    video_path = f"data/raw_videos/recordings/{video_id}/{CAMERA_ID}_rgb.mp4"
    pose_path = f"data/poses/poses@60fps/{video_id}.json"
    out_path = f"{OUTPUT_DIR}/{video_id}_{CAMERA_ID}_full.mp4"

    if not os.path.exists(video_path) or not os.path.exists(pose_path):
        print(f"⚠️ Skipping {video_id} - Missing MP4 or JSON.")
        return

    with open(pose_path, 'r') as f:
        pose_data = json.load(f)
    poses_by_frame = {frame['frame_index']: frame['landmarks'] for frame in pose_data}

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    print(f"\n▶️ Starting: {video_id} ({total_frames} total frames)")
    frame_count = 0
    
    # Process the entire video until it ends
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        landmarks = poses_by_frame.get(frame_count, {})
        for hand_id, color in [('0', (255, 150, 0)), ('1', (0, 255, 255))]:
            if hand_id in landmarks:
                points_3d = np.array(landmarks[hand_id], dtype=np.float32)
                points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)
                points_2d = points_2d.reshape(-1, 2)
                draw_skeleton(frame, points_2d, color=color)
        
        out.write(frame)
        frame_count += 1
        
        # Print progress every 1000 frames so it doesn't spam the terminal
        if frame_count % 1000 == 0:
            print(f"  ... rendered {frame_count} / {total_frames} frames")

    cap.release()
    out.release()
    print(f"✅ Finished: Saved to {out_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading universal camera matrices...")
    K, dist_coeffs, rvec, tvec = load_camera_matrices()

    # Find all downloaded recordings and grab the first 10
    recording_dirs = sorted(glob.glob("data/raw_videos/recordings/*"))
    video_ids = [os.path.basename(d) for d in recording_dirs][:10]
    
    print(f"Found {len(recording_dirs)} downloaded folders. Processing the first {len(video_ids)}...")

    for vid in video_ids:
        process_video(vid, K, dist_coeffs, rvec, tvec)
        
    print("\n🎉 Pipeline complete! 10 full-length videos have been rendered.")

if __name__ == "__main__":
    main()
