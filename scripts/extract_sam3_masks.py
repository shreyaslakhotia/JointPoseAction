import argparse
import glob
import os
import cv2
import numpy as np


def extract_binary_mask(image_path, visualize=False):
  """Extracts a solid binary mask from a SAM 3 overlay with a magenta/pink contour."""
  img_bgr = cv2.imread(image_path)
  if img_bgr is None:
    print(f'❌ Could not load: {image_path}')
    return None, None

  h, w = img_bgr.shape[:2]

  # 1. Convert to HSV color space to reliably isolate magenta / pink
  img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

  # Magenta/Pink typically sits in Hue 140-175 in OpenCV (H: 0-179, S: 0-255, V: 0-255)
  lower_magenta = np.array([135, 70, 100], dtype=np.uint8)
  upper_magenta = np.array([175, 255, 255], dtype=np.uint8)
  color_mask = cv2.inRange(img_hsv, lower_magenta, upper_magenta)

  # Secondary check in BGR: ensure high Red & Blue relative to Green
  b, g, r = cv2.split(img_bgr)
  bgr_condition = (r > 150) & (b > 100) & (g < 130) & (r > g + 40)
  combined_pink = color_mask & bgr_condition.astype(np.uint8) * 255

  # 2. Morphological close to bridge any 1-2px breaks in the contour line
  close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
  closed_contour = cv2.morphologyEx(combined_pink, cv2.MORPH_CLOSE, close_kernel)

  # 3. Find external contours of the magenta boundary
  contours, _ = cv2.findContours(
      closed_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  binary_mask = np.zeros((h, w), dtype=np.uint8)

  if contours:
    # Target the largest contour corresponding to the toy boundary
    valid_contours = [c for c in contours if cv2.contourArea(c) > 300]
    if valid_contours:
      target_contour = max(valid_contours, key=cv2.contourArea)
      # Fill the entire region enclosed by the boundary solid with 255
      cv2.drawContours(
          binary_mask, [target_contour], -1, 255, thickness=cv2.FILLED
      )

      # Optional morphological smoothing on the filled mask
      smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
      binary_mask = cv2.morphologyEx(
          binary_mask, cv2.MORPH_CLOSE, smooth_kernel
      )
    else:
      print(f'⚠️ Warning: Contours found but below area threshold for {image_path}')
  else:
    print(f'⚠️ Warning: No magenta contour detected in {image_path}')

  # 4. Create optional verification overlay (50% red fill with green outline)
  overlay = None
  if visualize:
    overlay = img_bgr.copy()
    colored_layer = np.zeros_like(img_bgr)
    colored_layer[binary_mask == 255] = [0, 0, 255]  # Red in BGR
    overlay = cv2.addWeighted(overlay, 0.75, colored_layer, 0.45, 0)
    out_contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, out_contours, -1, (0, 255, 0), 2)

  return binary_mask, overlay


def main():
  parser = argparse.ArgumentParser(
      description='Convert SAM 3 magenta contour visualizations to solid binary masks.'
  )
  parser.add_argument(
      '--input_dir',
      type=str,
      default='/home/mlda/shreyas_projects/JointPoseAction/experiments/sam2_test/sam3_segmentations',
      help='Directory containing SAM 3 output images.',
  )
  parser.add_argument(
      '--output_dir',
      type=str,
      default='/home/mlda/shreyas_projects/JointPoseAction/experiments/sam2_test/sam3_binary_masks',
      help='Directory to save lossless binary masks.',
  )
  parser.add_argument(
      '--visualize',
      type=str,
      choices=['yes', 'no'],
      default='yes',
      help="Generate verification overlays in 'mask_overlays/' (yes/no).",
  )
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  overlay_dir = os.path.join(args.output_dir, 'overlays')
  if args.visualize == 'yes':
    os.makedirs(overlay_dir, exist_ok=True)

  # Find all input PNG and JPG files
  image_paths = sorted(
      glob.glob(os.path.join(args.input_dir, '*.png'))
      + glob.glob(os.path.join(args.input_dir, '*.jpg'))
  )
  if not image_paths:
    print(f'❌ No images found in {args.input_dir}')
    return

  print(f'Found {len(image_paths)} SAM 3 visualizations to process...')

  for img_path in image_paths:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    cam_id = base_name.split('_')[0]

    mask, overlay = extract_binary_mask(
        img_path, visualize=(args.visualize == 'yes')
    )
    if mask is None:
      continue

    # Save 1-channel lossless binary PNG
    out_mask_path = os.path.join(args.output_dir, f'{cam_id}_mask.png')
    cv2.imwrite(out_mask_path, mask)

    # Save visual verification overlay if requested
    if overlay is not None:
      out_overlay_path = os.path.join(overlay_dir, f'{cam_id}_overlay.jpg')
      cv2.imwrite(out_overlay_path, overlay)

    print(
        f'✅ Generated binary mask: {os.path.basename(out_mask_path)} (Foreground'
        f' pixels: {np.count_nonzero(mask):,})'
    )

  print(f'\n🎉 Extraction complete! Binary masks saved to: {args.output_dir}')
  if args.visualize == 'yes':
    print(f'👁️ Verification overlays saved to: {overlay_dir}')


if __name__ == '__main__':
  main()
