"""
Test script for Step 1: Video Preprocessing
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append('src')

from preprocessing.video_processor import VideoPreprocessor, test_preprocessing

def main():
    print("="*60)
    print("TESTING STEP 1: VIDEO PREPROCESSING")
    print("="*60)
    
    # Run the test
    test_preprocessing()

if __name__ == "__main__":
    main()
