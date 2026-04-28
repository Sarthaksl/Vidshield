import os
import cv2
from dct_fingerprint_redundant_v2 import generate_fingerprint, embed_fingerprint

PROTECTED_DIR = "./protected_images"
OUTPUT_DIR = "./final_protected"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(PROTECTED_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(PROTECTED_DIR, filename)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping {filename} (cannot read)")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image_id = f"user123_{filename}"
    fingerprint = generate_fingerprint(image_id)

    watermarked = embed_fingerprint(img, fingerprint)

    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(
        output_path,
        cv2.cvtColor(watermarked, cv2.COLOR_RGB2BGR)
    )

    print(f"Embedded fingerprint in: {filename}")
