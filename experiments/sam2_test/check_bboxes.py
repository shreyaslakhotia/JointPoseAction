import cv2
import os
import glob

# Your defined bounding boxes [x_min, y_min, x_max, y_max]
bboxes = {
    "C10095": [734, 437, 1100, 642],
    "C10115": [743, 343, 1006, 606],
    "C10118": [721, 650, 1178, 964],
    "C10119": [836, 505, 1103, 750],
    "C10379": [1032, 600, 1367, 1000],
    "C10390": [786, 469, 1053, 680],
    "C10395": [750, 600, 1100, 1100],
    "C10404": [1035, 300, 1314, 700]
}



# Create a folder for the output images
output_dir = "bbox_validation"
os.makedirs(output_dir, exist_ok=True)

print("Drawing bounding boxes...")

for cam, bbox in bboxes.items():
    # Grab the first available frame for this camera
    frame_files = glob.glob(f"frames/{cam}_*.jpg")
    if not frame_files:
        print(f"⚠️ Could not find frames for {cam}")
        continue
    
    first_frame = frame_files[0]
    img = cv2.imread(first_frame)
    
    xmin, ymin, xmax, ymax = bbox
    
    # Draw a bright green rectangle (Thickness=4)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
    
    # Save the visual validation image
    out_path = os.path.join(output_dir, f"{cam}_bbox.jpg")
    cv2.imwrite(out_path, img)
    print(f"✅ Saved: {out_path}")

print("\nDone! Check the 'bbox_validation' folder.")
