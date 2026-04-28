import os
import json
import hashlib
import cv2
from dct_fingerprint_redundant_v2 import extract_fingerprint

# ======================
# PATH CONFIG
# ======================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANIFEST_DIR = os.path.join(BASE_DIR, "c2pa_manifests")
IMAGE_DIR = os.path.join(BASE_DIR, "final_protected", "class_dummy")

# ======================
# VERIFY ONE IMAGE CHAIN
# ======================

def verify_chain_for_image(manifest_path):

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    asset_name = manifest["asset_name"]
    image_path = os.path.join(IMAGE_DIR, asset_name)

    if not os.path.exists(image_path):
        print(f"⚠️ Image missing: {asset_name}")
        return

    chain = manifest.get("fingerprint_chain", [])

    if len(chain) == 0:
        print(f"❌ No fingerprint chain in {asset_name}")
        return

    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract fingerprint
    fp = extract_fingerprint(img)
    fp_bits = "".join(map(str, fp)).encode()
    current_hash = hashlib.sha256(fp_bits).hexdigest()

    # Check last hash matches current image
    last_hash = chain[-1]["hash"]

    if current_hash == last_hash:
        print(f"✅ VERIFIED: {asset_name}")
    else:
        print(f"❌ TAMPERED: {asset_name}")

    # ======================
    # VERIFY CHAIN LINKING
    # ======================

    chain_valid = True
    for i in range(1, len(chain)):
        prev_hash = chain[i-1]["hash"]
        curr_hash = chain[i]["hash"]

        expected = hashlib.sha256(prev_hash.encode() + curr_hash.encode()).hexdigest()

        if expected != curr_hash:
            chain_valid = False
            print(f"❌ Chain broken at version {i} for {asset_name}")
            break

    if chain_valid:
        print(f"🔗 Chain intact for {asset_name}")


# ======================
# RUN FOR ALL MANIFESTS
# ======================

for file in os.listdir(MANIFEST_DIR):
    if file.endswith(".json"):
        verify_chain_for_image(os.path.join(MANIFEST_DIR, file))
