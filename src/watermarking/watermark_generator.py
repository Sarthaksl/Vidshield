import numpy as np
import hashlib
from cryptography.fernet import Fernet
import os

class WatermarkGenerator:
    def __init__(self, watermark_length=64):
        self.watermark_length = watermark_length

    def generate_watermark(self, video_path):
        """
        Generate a binary watermark using SHA-256 hash of the video file.
        """
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        hash_digest = hashlib.sha256(video_bytes).hexdigest()
        # Convert hash to binary string, then to array of 0/1
        binary_str = bin(int(hash_digest, 16))[2:].zfill(256)[:self.watermark_length]
        watermark = np.array([int(b) for b in binary_str], dtype=np.uint8)
        return watermark

    def generate_key(self):
        """
        Generate a secret key for encryption.
        """
        return Fernet.generate_key()

    def encrypt_watermark(self, watermark, key):
        """
        Encrypt the watermark using Fernet symmetric encryption.
        """
        f = Fernet(key)
        watermark_bytes = bytes(watermark)
        encrypted = f.encrypt(watermark_bytes)
        return encrypted

    def save_watermark_and_key(self, encrypted_watermark, key, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "encrypted_watermark.bin"), "wb") as f:
            f.write(encrypted_watermark)
        with open(os.path.join(save_dir, "encryption_key.key"), "wb") as f:
            f.write(key)
        print(f"✅ Watermark and key saved in {save_dir}")

if __name__ == "__main__":
    video_path = "../../data/input_videos/sample_video.mp4"  # Adjust as needed
    save_dir = "../../data/processed_frames/"
    wg = WatermarkGenerator(watermark_length=64)
    watermark = wg.generate_watermark(video_path)
    np.save("../../data/processed_frames/watermark_bits.npy", watermark)
    print("✅ Watermark bits saved as watermark_bits.npy")
    key = wg.generate_key()
    encrypted_watermark = wg.encrypt_watermark(watermark, key)
    wg.save_watermark_and_key(encrypted_watermark, key, save_dir)
    print("Watermark (binary):", watermark)
    print("Encryption key:", key)
