import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def generate_heatmap(extracted_bits, original_bits, attn_map, frame_shape):
    """
    Create a heatmap showing differences on the stable pixels only.
    Args:
        extracted_bits (np.ndarray): Extracted watermark bits array.
        original_bits (np.ndarray): Original watermark bits array.
        attn_map (np.ndarray): Attention mask for stable pixels [H, W].
        frame_shape (tuple): Shape (H, W) of the frame.
    Returns:
        heatmap (np.ndarray): Heatmap image highlighting differences.
    """
    heatmap = np.zeros(frame_shape, dtype=np.uint8)
    stable_indices = np.where(attn_map.flatten() == 1)[0]

    # Loop only within range of watermark bits and stable pixels
    num_bits = min(len(extracted_bits), len(original_bits), len(stable_indices))
    for i in range(num_bits):
        idx = stable_indices[i]
        bit_diff = abs(int(extracted_bits[i]) - int(original_bits[i]))
        # Map bit difference (0 or 1) to 0 or 255
        heatmap.flat[idx] = bit_diff * 255

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color

def save_heatmaps_for_frames(
    tampered_frames_folder, 
    attn_maps_path, 
    original_watermark_bits, 
    extracted_bits_list, 
    output_folder
):
    tampered_frames_folder = Path(tampered_frames_folder)
    attn_maps = np.load(attn_maps_path)
    frame_files = sorted([f for f in tampered_frames_folder.iterdir() if f.suffix.lower() == ".jpg"])
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        attn_map = attn_maps[i]
        extracted_bits = extracted_bits_list[i]

        heatmap = generate_heatmap(extracted_bits, original_watermark_bits, attn_map, attn_map.shape)
        blended = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

        out_path = output_folder / frame_file.name
        cv2.imwrite(str(out_path), blended)

    print(f"✅ Heatmaps saved to {output_folder}")

if __name__ == "__main__":
    from watermark_verifier import WatermarkVerifier
    import numpy as np

    watermark_path = "../../data/processed_frames/watermark_bits.npy"
    tampered_frames_folder = "../../data/output_videos/tampered_frames/"
    attn_maps_path = "../../data/processed_frames/attention_maps.npy"
    heatmap_output_folder = "../../data/output_videos/heatmaps/"

    original_watermark_bits = np.load(watermark_path)
    verifier = WatermarkVerifier(original_watermark_bits)

    tampered_frames = sorted(list(Path(tampered_frames_folder).glob("*.jpg")))
    attn_maps = np.load(attn_maps_path)

    extracted_bits_list = []
    for i, frame_file in enumerate(tampered_frames):
        frame = cv2.imread(str(frame_file))
        attn_map = attn_maps[i]
        bits = verifier.extract_from_frame(frame, attn_map)
        extracted_bits_list.append(bits)

    save_heatmaps_for_frames(
        tampered_frames_folder,
        attn_maps_path,
        original_watermark_bits,
        extracted_bits_list,
        heatmap_output_folder
    )
