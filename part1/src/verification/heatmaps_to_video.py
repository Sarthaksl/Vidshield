import cv2
from pathlib import Path

def images_to_video(images_folder, output_video_path, fps=10):
    images_folder = Path(images_folder)
    image_files = sorted([f for f in images_folder.iterdir() if f.suffix.lower() == '.jpg'])

    if not image_files:
        print("No images found in folder:", images_folder)
        return

    # Read first image to get size
    first_frame = cv2.imread(str(image_files[0]))
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    print(f"Creating video {output_video_path} from {len(image_files)} frames...")

    for img_file in image_files:
        frame = cv2.imread(str(img_file))
        video.write(frame)

    video.release()
    print(f"✅ Video saved: {output_video_path}")

if __name__ == "__main__":
    heatmaps_folder = "../../data/output_videos/heatmaps/"
    output_video = "../../data/output_videos/heatmap_visualization.mp4"
    images_to_video(heatmaps_folder, output_video, fps=10)
