import os
import cv2
import numpy as np
from dct_fingerprint_redundant_v2 import extract_fingerprint, generate_fingerprint

FINAL_DIR = "./final_protected"

for filename in os.listdir(FINAL_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(FINAL_DIR, filename)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping {filename} (cannot read)")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    extracted = extract_fingerprint(img)

    image_id = f"user123_{filename}"
    original = generate_fingerprint(image_id)

    accuracy = np.mean(extracted == original)

    print(f"{filename} → Fingerprint match accuracy: {accuracy:.4f}")
