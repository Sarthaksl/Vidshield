import os
import json
import hashlib
import cv2
from dct_fingerprint_redundant_v2 import extract_fingerprint

# Robust base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "final_protected", "class_dummy")
MANIFEST_DIR = os.path.join(BASE_DIR, "c2pa_manifests")

for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    manifest_path = os.path.join(
        MANIFEST_DIR, f"{filename}.c2pa.json"
    )

    if not os.path.exists(manifest_path):
        print(f"❌ No manifest found for {filename}")
        continue

    # Load manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    stored_hash = manifest["content_binding"]["hash"]

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image {filename}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract fingerprint again
    fingerprint = extract_fingerprint(img)
    current_bits = "".join(map(str, fingerprint))
    current_hash = hashlib.sha256(
        current_bits.encode()
    ).hexdigest()

    # Verify
    if current_hash == stored_hash:
        print(f"✅ VERIFIED: {filename} is authentic")
    else:
        print(f"❌ TAMPERED: {filename} provenance broken")

    print("   Stored hash :", stored_hash)
    print("   Current hash:", current_hash)
