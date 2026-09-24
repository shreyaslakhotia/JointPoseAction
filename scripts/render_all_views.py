import json
import cv2
import numpy as np
import os
import glob
import multiprocessing
import concurrent.futures

# --- Configuration ---
INPUT_DIR = "data/raw_videos/recordings"
OUTPUT_DIR = "output_full_videos"
POSE_DIR = "data/poses/poses@60fps"
# The universal calibration file contains the math for ALL 8 fixed cameras
CALIB_PATH = "assemblyhands-toolkit/calib/nimble_json_calib/nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345.json"

# Pre-load calibration so we don't read the hard drive 400 times
print("Loading master calibration file...")
with open(CALIB_PATH, 'r') as f:
    CALIB_DATA = json.load(f)

def get_camera_matrices(camera_id):
    """Dynamically fetches the exact K, R, T for the requested camera ID."""
    cam_data = None
    for item in CALIB_DATA:
        if item.get('Camera', {}).get('SerialNo') == f"{camera_id}_rgb":
            cam_data = item['Camera']
            break
            
    if not cam_data:
        raise ValueError(f"Could not find {camera_id} in calibration!")

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
    rvec, _ = cv2.Rodrigues(E[:3, :3])
    tvec = E[:3, 3].reshape(3, 1)
    
    return K, dist_coeffs, rvec, tvec

def draw_skeleton(frame, points_2d, color):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),       
        (0, 5), (5, 6), (6, 7), (7, 8),       
        (0, 9), (9, 10), (10, 11), (11, 12),  
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20) 
    ]
    if np.isnan(points_2d).any(): return
        
    points_2d = points_2d.astype(int)
    h, w = frame.shape[:2]
    
    for pt1, pt2 in edges:
        if pt1 < len(points_2d) and pt2 < len(points_2d):
            if (0 <= points_2d[pt1][0] <= w*2) and (0 <= points_2d[pt1][1] <= h*2):
                cv2.line(frame, tuple(points_2d[pt1]), tuple(points_2d[pt2]), color, 2)
            
    for pt in points_2d:
        if (0 <= pt[0] <= w*2) and (0 <= pt[1] <= h*2):
            cv2.circle(frame, tuple(pt), 4, (0, 0, 255), -1)

def process_single_video(task_args):
    """The worker function that handles one single video."""
    video_path, pose_path, out_path, camera_id = task_args
    
    if os.path.exists(out_path):
        return f"⏭️ Skipping (Already done): {os.path.basename(out_path)}"
        
    try:
        K, dist_coeffs, rvec, tvec = get_camera_matrices(camera_id)
    except Exception as e:
        return f"❌ Error loading matrices for {camera_id}: {e}"

    with open(pose_path, 'r') as f:
        pose_data = json.load(f)
    poses_by_frame = {frame['frame_index']: frame['landmarks'] for frame in pose_data}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        landmarks = poses_by_frame.get(frame_count, {})
        for hand_id, color in [('0', (255, 150, 0)), ('1', (0, 255, 255))]:
            if hand_id in landmarks:
                points_3d = np.array(landmarks[hand_id], dtype=np.float32)
                points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)
                draw_skeleton(frame, points_2d.reshape(-1, 2), color)
                
        out.write(frame)
        frame_count += 1
        
    cap.release()
    out.release()
    return f"✅ Finished: {os.path.basename(out_path)} ({frame_count} frames)"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tasks = []
    
    # 1. Build the task list and mirror the directory structure
    for video_dir in sorted(glob.glob(f"{INPUT_DIR}/*")):
        video_id = os.path.basename(video_dir)
        
        # Mirror the folder in output
        out_vid_dir = os.path.join(OUTPUT_DIR, video_id)
        os.makedirs(out_vid_dir, exist_ok=True)
        
        pose_path = os.path.join(POSE_DIR, f"{video_id}.json")
        if not os.path.exists(pose_path):
            continue
            
        # Add all 8 views to the task list
        for video_path in glob.glob(f"{video_dir}/*.mp4"):
            filename = os.path.basename(video_path)
            camera_id = filename.split('_')[0] # e.g., 'C10395'
            out_path = os.path.join(out_vid_dir, filename)
            
            tasks.append((video_path, pose_path, out_path, camera_id))
            
    print(f"📋 Total videos queued for processing: {len(tasks)}")
    
    # 2. Process in parallel (leaving 2 CPU cores free so the OS doesn't freeze)
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Starting {num_workers} parallel workers to churn through the videos...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(process_single_video, tasks):
            print(result)

    print("\n🎉 All 400 videos have been successfully rendered with 3D overlays!")

if __name__ == '__main__':
    main()
