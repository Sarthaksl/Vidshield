import numpy as np
import os
from pathlib import Path
import cv2

class AttentionMapGenerator:
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def load_frames_from_jpg(self, frames_folder):
        """
        Loads all .jpg frames from a folder into a numpy array.
        Assumes frames are named in order (e.g., frame_000001.jpg).
        Returns: numpy array of shape (N, H, W, 3), normalized to [0,1]
        """
        frames_folder = Path(frames_folder)
        frame_files = sorted([f for f in frames_folder.iterdir() if f.suffix.lower() == '.jpg'])
        frames = []
        for f in frame_files:
            img = cv2.imread(str(f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = img.astype(np.float32) / 255.0        # Normalize
            frames.append(img)
        frames = np.stack(frames, axis=0)
        print(f"Loaded {len(frames)} frames from {frames_folder}")
        return frames

    def compute_attention_maps(self, frames):
        N, H, W, C = frames.shape
        attention_maps = np.zeros((N, H, W), dtype=np.uint8)
        for i in range(1, N):
            diff = np.abs(frames[i] - frames[i - 1])
            diff_per_pixel = np.mean(diff, axis=2)
            mask = (diff_per_pixel < self.threshold).astype(np.uint8)
            attention_maps[i] = mask
        attention_maps[0] = np.ones((H, W), dtype=np.uint8)
        return attention_maps

    def save_attention_maps(self, maps, save_path):
        save_path = Path(save_path)
        np.save(save_path, maps)
        print(f"✅ Attention maps saved to: {save_path}")

if __name__ == "__main__":
    frames_folder = '../../data/processed_frames/'  # Adjust if needed
    save_path = '../../data/processed_frames/attention_maps.npy'

    ag = AttentionMapGenerator(threshold=0.05)
    frames = ag.load_frames_from_jpg(frames_folder)
    attn_maps = ag.compute_attention_maps(frames)
    ag.save_attention_maps(attn_maps, save_path)
    print(f"Attention maps shape: {attn_maps.shape}")
    print(f"Sample map (frame 5, unique values): {np.unique(attn_maps[5])}")
