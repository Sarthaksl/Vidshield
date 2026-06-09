# src/pipeline_orchestrator.py
import os
from pathlib import Path
from src.preprocessing.video_processor import VideoPreprocessor
from src.attention.attention_map_generator import AttentionMapGenerator
from src.watermarking.watermark_generator import WatermarkGenerator
from src.watermarking.watermark_embedder import WatermarkEmbedder
from src.attack.simulate_tampering import simulate_tamper
from src.metadata.store_metadata import store_metadata
from src.verification.watermark_verifier import WatermarkVerifier
from src.verification.heatmap_generator import generate_heatmap, save_heatmaps_for_frames
from src.verification.heatmaps_to_video import images_to_video
import numpy as np
import cv2

def process_pipeline(video_path, uploader):
    processed_frames_folder = Path("data/processed_frames/")
    output_videos_folder = Path("data/output_videos/")
    processed_frames_folder.mkdir(parents=True, exist_ok=True)
    output_videos_folder.mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocessing
    preprocessor = VideoPreprocessor(target_size=(224, 224))
    frames = preprocessor.extract_frames(
        video_path=str(video_path),
        output_folder=str(processed_frames_folder),
        max_frames=100
    )

    # Step 2: Attention
    attention_maker = AttentionMapGenerator(threshold=0.05)
    attn_maps = attention_maker.compute_attention_maps(frames)
    attn_maps_path = processed_frames_folder / "attention_maps.npy"
    attention_maker.save_attention_maps(attn_maps, attn_maps_path)

    # Step 3: Watermark & Key
    wg = WatermarkGenerator(watermark_length=64)
    watermark = wg.generate_watermark(video_path)
    np.save(processed_frames_folder / "watermark_bits.npy", watermark)
    key = wg.generate_key()
    encrypted_watermark = wg.encrypt_watermark(watermark, key)
    wg.save_watermark_and_key(encrypted_watermark, key, processed_frames_folder)

    # Step 4: Watermark Embedding
    embedder = WatermarkEmbedder(watermark)
    watermarked_frames_folder = output_videos_folder / "watermarked_frames"
    watermarked_frames_folder.mkdir(parents=True, exist_ok=True)
    embedder.embed_watermark(str(processed_frames_folder), attn_maps_path, str(watermarked_frames_folder))

    # Step 5: Metadata Storage
    video_metadata_path = processed_frames_folder / "video_metadata.json"
    store_metadata(video_path, uploader, processed_frames_folder / "watermark_bits.npy", video_metadata_path)

    # Step 6: Tampering
    tampered_frames_folder = output_videos_folder / "tampered_frames"
    simulate_tamper(str(watermarked_frames_folder), str(tampered_frames_folder), method="rectangle")

    # ----------- NEW: Assemble tampered frames into video -----------
    tampered_video_path = output_videos_folder / "tampered.mp4"
    images_to_video(str(tampered_frames_folder), str(tampered_video_path), fps=10)

    # Step 7: Verification
    verifier = WatermarkVerifier(watermark)
    is_verified = verifier.verify(str(tampered_frames_folder), attn_maps_path)
    verdict = "VERIFIED" if is_verified else "TAMPERED"

    # For heatmap generation (per frame extraction)
    tampered_frame_files = sorted(list(tampered_frames_folder.glob("*.jpg")))
    original_watermark_bits = np.load(processed_frames_folder / "watermark_bits.npy")
    attn_maps_loaded = np.load(attn_maps_path)
    extracted_bits_list = []
    for i, frame_file in enumerate(tampered_frame_files):
        frame = cv2.imread(str(frame_file))
        attn_map = attn_maps_loaded[i]
        bits = verifier.extract_from_frame(frame, attn_map)
        extracted_bits_list.append(bits)
    heatmaps_folder = output_videos_folder / "heatmaps"
    save_heatmaps_for_frames(str(tampered_frames_folder), attn_maps_path, original_watermark_bits, extracted_bits_list, str(heatmaps_folder))

    # Step 8: Heatmap Video
    heatmap_video_path = output_videos_folder / "heatmap_visualization.mp4"
    images_to_video(str(heatmaps_folder), str(heatmap_video_path), fps=10)

    # For display in frontend (return filenames/paths)
    return {
        "original_video": str(video_path),
        "tampered_video": str(tampered_video_path),
        "heatmap_video": str(heatmap_video_path),
        "status": verdict
    }
