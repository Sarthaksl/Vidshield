import cv2
import numpy as np
import hashlib

# =====================
# CONFIG
# =====================
BLOCK = 8
ALPHA = 12.0
FREQ_POS = (4, 3)
BITS = 32
REPEAT = 12
SEED = 2026   # secret key

# =====================
# FINGERPRINT
# =====================
def generate_fingerprint(image_id: str, bits=BITS):
    h = hashlib.sha256(image_id.encode()).hexdigest()
    binary = bin(int(h, 16))[2:].zfill(256)
    return np.array([int(b) for b in binary[:bits]])

# =====================
# DETERMINISTIC BLOCK MAP
# =====================
def get_block_map(h, w):
    blocks = []
    for i in range(0, h - BLOCK, BLOCK):
        for j in range(0, w - BLOCK, BLOCK):
            blocks.append((i, j))
    return blocks

# =====================
# EMBED (DETERMINISTIC)
# =====================
def embed_fingerprint(image, fingerprint):
    img = image.astype(np.float32)
    h, w, _ = img.shape

    blocks = get_block_map(h, w)
    np.random.seed(SEED)
    np.random.shuffle(blocks)

    required = BITS * REPEAT
    assert required <= len(blocks), "Image too small for watermark"

    idx = 0
    for bit_idx, bit in enumerate(fingerprint):
        for _ in range(REPEAT):
            i, j = blocks[idx]
            idx += 1

            c = (bit_idx + idx) % 3   # deterministic channel
            block = img[i:i+BLOCK, j:j+BLOCK, c]
            dct = cv2.dct(block)

            dct[FREQ_POS] += ALPHA if bit == 1 else -ALPHA
            img[i:i+BLOCK, j:j+BLOCK, c] = cv2.idct(dct)

    return np.clip(img, 0, 255).astype(np.uint8)

# =====================
# EXTRACT (MATCHING MAP)
# =====================
def extract_fingerprint(image, bits=BITS):
    img = image.astype(np.float32)
    h, w, _ = img.shape

    blocks = get_block_map(h, w)
    np.random.seed(SEED)
    np.random.shuffle(blocks)

    votes = [[] for _ in range(bits)]

    idx = 0
    for bit_idx in range(bits):
        for _ in range(REPEAT):
            i, j = blocks[idx]
            idx += 1

            c = (bit_idx + idx) % 3
            block = img[i:i+BLOCK, j:j+BLOCK, c]
            dct = cv2.dct(block)
            votes[bit_idx].append(dct[FREQ_POS])

    extracted = [1 if np.mean(v) > 0 else 0 for v in votes]
    return np.array(extracted)
