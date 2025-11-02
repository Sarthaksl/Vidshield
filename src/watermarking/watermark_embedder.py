import numpy as np
import cv2
from pathlib import Path

class WatermarkEmbedder:
    def __init__(self, watermark_bits):
        self.watermark_bits = watermark_bits

    def embed_in_frame(self, frame, attn_map):
        """
        Embed watermark bits into the frame using the attention map.
        Only modifies the blue channel's LSB of stable pixels.
        """
        h, w, _ = frame.shape
        flat_frame = frame.reshape(-1, 3)
        flat_attn = attn_map.flatten()
        stable_indices = np.where(flat_attn == 1)[0]
        num_bits = len(self.watermark_bits)
        # If more stable pixels than bits, repeat watermark
        bits_to_embed = np.resize(self.watermark_bits, len(stable_indices))
        # Embed bits
        for idx, bit in zip(stable_indices, bits_to_embed):
            # Set LSB of blue channel
           flat_frame[idx, 0] = (flat_frame[idx, 0] & 0xFE) | bit

        return flat_frame.reshape(h, w, 3)

    def embed_watermark(self, frames_folder, attn_maps_path, output_folder):
        """
        Embed watermark into all frames in frames_folder using attention maps.
        """
        frames_folder = Path(frames_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        attn_maps = np.load(attn_maps_path)
        frame_files = sorted([f for f in frames_folder.iterdir() if f.suffix.lower() == '.jpg'])
        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            attn_map = attn_maps[i]
            watermarked = self.embed_in_frame(frame, attn_map)
            out_path = output_folder / frame_file.name
            cv2.imwrite(str(out_path), watermarked)
        print(f"✅ Watermarked frames saved to {output_folder}")

if __name__ == "__main__":
    # Load watermark bits (unencrypted for embedding)
    # For demo, load from previous step or use a test array
    watermark_bits = np.load("../../data/processed_frames/watermark_bits.npy")  # Save your watermark as .npy in previous step
    frames_folder = "../../data/processed_frames/"
    attn_maps_path = "../../data/processed_frames/attention_maps.npy"
    output_folder = "../../data/output_videos/watermarked_frames/"
    embedder = WatermarkEmbedder(watermark_bits)
    embedder.embed_watermark(frames_folder, attn_maps_path, output_folder)
