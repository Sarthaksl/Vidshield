import numpy as np
import cv2
from pathlib import Path

class WatermarkVerifier:
    def __init__(self, original_watermark_bits):
        self.original_watermark_bits = original_watermark_bits

    def extract_from_frame(self, frame, attn_map):
        """
        Extract bits from the blue channel's LSB of stable pixels.
        """
        h, w, _ = frame.shape
        flat_frame = frame.reshape(-1, 3)
        flat_attn = attn_map.flatten()
        stable_indices = np.where(flat_attn == 1)[0]

        extracted_bits = []
        for idx in stable_indices:
            bit = flat_frame[idx, 0] & 1  # LSB of blue channel
            extracted_bits.append(bit)
        # Truncate or pad extracted bits to original watermark length
        extracted_bits = extracted_bits[:len(self.original_watermark_bits)]
        return np.array(extracted_bits, dtype=np.uint8)

    def verify(self, tampered_frames_folder, attn_maps_path):
        tampered_frames_folder = Path(tampered_frames_folder)
        attn_maps = np.load(attn_maps_path)
        frame_files = sorted([f for f in tampered_frames_folder.iterdir() if f.suffix.lower() == '.jpg'])

        differences = []
        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            attn_map = attn_maps[i]
            extracted_bits = self.extract_from_frame(frame, attn_map)
            diff = np.sum(self.original_watermark_bits != extracted_bits) / len(self.original_watermark_bits)
            differences.append(diff)

        avg_difference = np.mean(differences)
        print(f"Average bit difference between original and extracted watermark: {avg_difference:.4f}")

        threshold = 0.1  # Define acceptable difference threshold
        if avg_difference > threshold:
            print("⚠️ Video is likely TAMPERED")
            return False
        else:
            print("✅ Video is VERIFIED as authentic")
            return True

if __name__ == "__main__":
    watermark_path = "../../data/processed_frames/watermark_bits.npy"
    tampered_frames_folder = "../../data/output_videos/tampered_frames/"
    attn_maps_path = "../../data/processed_frames/attention_maps.npy"

    original_watermark_bits = np.load(watermark_path)
    verifier = WatermarkVerifier(original_watermark_bits)
    result = verifier.verify(tampered_frames_folder, attn_maps_path)
