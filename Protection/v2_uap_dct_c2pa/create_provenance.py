import os
import json
import hashlib
import cv2
from dct_fingerprint_redundant_v2 import extract_fingerprint
import sys 

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Robust base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "final_protected", "class_dummy")
OUTPUT_DIR = os.path.join(BASE_DIR, "c2pa_manifests")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(INPUT_DIR, filename)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping {filename} (cannot read)")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract robust fingerprint
    fingerprint = extract_fingerprint(img)

    # Convert fingerprint to hash
    fingerprint_bits = "".join(map(str, fingerprint))
    fingerprint_hash = hashlib.sha256(
        fingerprint_bits.encode()
    ).hexdigest()

    # Build C2PA-style manifest
    manifest = {
        "c2pa_version": "1.3",
        "asset_type": "image",
        "asset_name": filename,
        "content_binding": {
            "method": "robust_fingerprint_hash",
            "hash_alg": "sha256",
            "hash": fingerprint_hash
        },
        "assertions": {
            "ai_protection": ["UAP", "DCT Fingerprint", "Neural Watermark"],
            "creator": "Demo User",
            "timestamp": "2026-02-10T10:30:00Z"
        }
    }

    manifest_path = os.path.join(
        OUTPUT_DIR, f"{filename}.c2pa.json"
    )

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"✅ C2PA provenance created for {filename}")
    print("   Fingerprint hash:", fingerprint_hash)
