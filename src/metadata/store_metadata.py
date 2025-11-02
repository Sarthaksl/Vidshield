import hashlib
import json
from pathlib import Path
import numpy as np

def compute_file_hash(file_path):
    # Compute SHA256 hash of a file
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()

def store_metadata(
    video_path,
    uploader,
    watermark_bits_path,
    output_metadata_path
):
    # Compute video fingerprint
    video_hash = compute_file_hash(video_path)
    # Load watermark bits and convert to string for human readability/logging
    watermark_bits = np.load(watermark_bits_path)
    watermark_str = ''.join(str(b) for b in watermark_bits.tolist())

    # Build metadata dict
    metadata = {
        "video_file": str(Path(video_path).name),
        "uploader": uploader,
        "video_hash": video_hash,
        "watermark_reference": watermark_str
    }

    # Write JSON file
    with open(output_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved to {output_metadata_path}")

if __name__ == "__main__":
    # Example usage:
    video_path = "../../data/input_videos/sample_video.mp4"
    uploader = "user@example.com"
    watermark_bits_path = "../../data/processed_frames/watermark_bits.npy"
    output_metadata_path = "../../data/processed_frames/video_metadata.json"
    store_metadata(video_path, uploader, watermark_bits_path, output_metadata_path)
