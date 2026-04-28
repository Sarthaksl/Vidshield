import cv2
import numpy as np
from pathlib import Path
import random

def simulate_tamper(src_folder, dst_folder, method="rectangle", tamper_ratio=0.15):
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    frame_files = sorted([f for f in src_folder.iterdir() if f.suffix.lower() == ".jpg"])

    print(f"Simulating tampering on {len(frame_files)} frames with method: {method}")

    for i, file in enumerate(frame_files):
        img = cv2.imread(str(file))
        h, w, _ = img.shape

        if method == "rectangle":
            # Overwrite a random rectangle region
            box_h, box_w = int(h * tamper_ratio), int(w * tamper_ratio)
            top_left = (random.randint(0, h - box_h), random.randint(0, w - box_w))
            # Set the region to gray
            img[top_left[0]:top_left[0]+box_h, top_left[1]:top_left[1]+box_w] = (127, 127, 127)
        elif method == "blur":
            # Blur a random region
            box_h, box_w = int(h * tamper_ratio), int(w * tamper_ratio)
            top_left = (random.randint(0, h - box_h), random.randint(0, w - box_w))
            roi = img[top_left[0]:top_left[0]+box_h, top_left[1]:top_left[1]+box_w]
            img[top_left[0]:top_left[0]+box_h, top_left[1]:top_left[1]+box_w] = cv2.GaussianBlur(roi, (15, 15), 0)
        else:
            raise ValueError("Unknown method: " + method)

        out_path = dst_folder / file.name
        cv2.imwrite(str(out_path), img)
    print(f"✅ Tampered frames saved in {dst_folder}")

if __name__ == "__main__":
    # Use your watermarked frames as input
    src_folder = "../../data/output_videos/watermarked_frames/"
    dst_folder = "../../data/output_videos/tampered_frames/"
    simulate_tamper(src_folder, dst_folder, method="rectangle")  # Can switch to "blur" if desired
