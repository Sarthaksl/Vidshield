import os
import json
import hashlib
import cv2
from dct_fingerprint_redundant_v2 import extract_fingerprint

# Robust base directory (works regardless of run location)
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

    chain = manifest.get("fingerprint_chain", [])

    if len(chain) == 0:
        print(f"⚠️ No chain found in manifest for {filename}")
        continue

    prev_hash = chain[-1]["hash"]

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image {filename}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract fingerprint
    fp = extract_fingerprint(img)
    fp_bits = "".join(map(str, fp)).encode()

    # Compute new chained hash
    new_hash = hashlib.sha256(
        prev_hash.encode() + fp_bits
    ).hexdigest()

    chain.append({
        "version": len(chain),
        "hash": new_hash,
        "action": "image_modified"
    })

    manifest["fingerprint_chain"] = chain

    # Save updated manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"🔗 Fingerprint chain updated for {filename}")
