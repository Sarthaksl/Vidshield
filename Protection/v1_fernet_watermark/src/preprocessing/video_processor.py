import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

class VideoPreprocessor:
    """
    Step 1: Video Preprocessing for Deepfake Prevention
    
    This class handles:
    - Breaking video into frames
    - Resizing frames to standard size
    - Normalizing pixel values (0-1 range)
    """
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize preprocessor
        
        Args:
            target_size (tuple): Target frame size (width, height)
        """
        self.target_size = target_size
        print(f"Video Preprocessor initialized with target size: {target_size}")
    
    def extract_frames(self, video_path, output_folder=None, max_frames=None):
        """
        Extract and preprocess frames from video
        
        Args:
            video_path (str): Path to input video
            output_folder (str): Where to save processed frames (optional)
            max_frames (int): Maximum number of frames to extract (optional)
        
        Returns:
            numpy.ndarray: Array of preprocessed frames
        """
        # Convert to Path object for easier handling
        video_path = Path(video_path)
        
        # Check if video exists
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video Info:")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - FPS: {fps}")
        print(f"  - Total frames: {total_frames}")
        
        # Limit frames if specified
        frames_to_process = min(total_frames, max_frames) if max_frames else total_frames
        
        # Storage for processed frames
        processed_frames = []
        
        # Create output folder if specified
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
        
        # Process frames with progress bar
        with tqdm(total=frames_to_process, desc="Processing frames") as pbar:
            frame_count = 0
            
            while frame_count < frames_to_process:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Step 1a: Resize frame to target size
                resized_frame = cv2.resize(frame, self.target_size)
                
                # Step 1b: Convert BGR to RGB (OpenCV uses BGR by default)
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                # Step 1c: Normalize pixel values to 0-1 range
                normalized_frame = rgb_frame.astype(np.float32) / 255.0
                
                # Store processed frame
                processed_frames.append(normalized_frame)
                
                # Save frame if output folder specified
                if output_folder:
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = output_folder / frame_filename
                    
                    # Convert back to 0-255 range for saving
                    save_frame = (normalized_frame * 255).astype(np.uint8)
                    save_frame_bgr = cv2.cvtColor(save_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(frame_path), save_frame_bgr)
                
                frame_count += 1
                pbar.update(1)
        
        # Clean up
        cap.release()
        
        # Convert to numpy array
        processed_frames = np.array(processed_frames)
        
        print(f"✅ Preprocessing complete!")
        print(f"   - Processed {len(processed_frames)} frames")
        print(f"   - Output shape: {processed_frames.shape}")
        print(f"   - Pixel value range: {processed_frames.min():.3f} - {processed_frames.max():.3f}")
        
        return processed_frames
    
    def save_processed_video(self, frames, output_path, fps=30):
        """
        Save processed frames back to video file
        
        Args:
            frames (numpy.ndarray): Processed frames array
            output_path (str): Output video path
            fps (float): Frames per second
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get frame dimensions
        height, width = frames.shape[1:3]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"Saving video to: {output_path}")
        
        for frame in tqdm(frames, desc="Saving frames"):
            # Convert from normalized float to uint8
            frame_uint8 = (frame * 255).astype(np.uint8)
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"✅ Video saved successfully!")

# Example usage function
def test_preprocessing():
    """Test the preprocessing pipeline"""
    
    # Initialize preprocessor
    processor = VideoPreprocessor(target_size=(224, 224))
    
    # Test with a sample video (you'll need to add a video file)
    video_path = "data/input_videos/sample_video.mp4"
    
    # Check if test video exists
    if not Path(video_path).exists():
        print("❌ Test video not found!")
        print(f"Please add a sample video to: {video_path}")
        return
    
    try:
        # Extract and process frames
        frames = processor.extract_frames(
            video_path=video_path,
            output_folder="data/processed_frames/",
            max_frames=100  # Process first 100 frames for testing
        )
        
        # Save processed video
        processor.save_processed_video(
            frames=frames,
            output_path="data/output_videos/preprocessed_sample.mp4",
            fps=30
        )
        
        print("\n🎉 Step 1 (Preprocessing) completed successfully!")
        print("Next: Implement Step 2 (Attention mechanism)")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")

if __name__ == "__main__":
    test_preprocessing()
