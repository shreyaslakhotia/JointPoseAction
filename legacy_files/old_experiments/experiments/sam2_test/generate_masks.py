import os
import glob
import cv2
import torch
import numpy as np
import argparse

# Import SAM-2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate SAM-2 masks with optional visualization.")
    parser.add_argument("--visualize", type=str, choices=["yes", "no"], default="no", 
                        help="Generate colored overlays for verification (yes/no)")
    args = parser.parse_args()

    # --- Configuration ---
    CHECKPOINT = "/home/mlda/shreyas_projects/sam2/checkpoints/sam2.1_hiera_large.pt"
    MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    FRAMES_DIR = "frames"
    OUTPUT_DIR = "masks"
    OVERLAY_DIR = "mask_overlays"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.visualize == "yes":
        os.makedirs(OVERLAY_DIR, exist_ok=True)

    # Your refined bounding boxes
    bboxes = {
        "C10095": [734, 437, 1100, 642],
        "C10115": [743, 343, 1006, 606],
        "C10118": [721, 650, 1178, 964],
        "C10119": [836, 505, 1103, 750],
        "C10379": [1032, 600, 1367, 1000],
        "C10390": [786, 469, 1053, 680],
        "C10395": [750, 600, 1100, 1050],
        "C10404": [1035, 300, 1314, 700]
    }

    print("Loading SAM-2 Model into GPU memory...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    print("Starting Mask Generation...")

    for cam, bbox in bboxes.items():
        frame_files = sorted(glob.glob(f"{FRAMES_DIR}/{cam}_*.jpg"))
        input_box = np.array(bbox)
        
        for frame_path in frame_files:
            filename = os.path.basename(frame_path)
            
            # Load image (OpenCV loads BGR, SAM-2 expects RGB)
            img_bgr = cv2.imread(frame_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 1. Pass the image to the model
            predictor.set_image(img_rgb)
            
            # 2. Prompt the model with your bounding box
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :], 
                multimask_output=False 
            )
            
            mask_bool = np.asarray(masks[0]).squeeze().astype(bool)
            mask_img = (mask_bool * 255).astype(np.uint8)
            
            # Save raw mask
            out_path = os.path.join(OUTPUT_DIR, filename.replace('.jpg', '_mask.png'))
            cv2.imwrite(out_path, mask_img)
            
            # 3. Handle optional visualization
            if args.visualize == "yes":
                colored_mask = np.zeros_like(img_bgr)
                colored_mask[mask_bool] = [0, 0, 255] # Red in BGR
                
                # Blend the image and the colored mask (50% transparency)
                overlay = cv2.addWeighted(img_bgr, 0.8, colored_mask, 0.5, 0)
                
                # Draw a crisp green outline around the edges
                contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                
                overlay_path = os.path.join(OVERLAY_DIR, filename.replace('.jpg', '_overlay.jpg'))
                cv2.imwrite(overlay_path, overlay)
            
            print(f"✅ Generated mask for {filename} (Confidence: {scores[0]:.2f})")

    print(f"\n🎉 Done! All masks saved to {OUTPUT_DIR}/")
    if args.visualize == "yes":
        print(f"👁️ Visual overlays saved to {OVERLAY_DIR}/")

if __name__ == "__main__":
    main()
