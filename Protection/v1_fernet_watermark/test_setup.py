import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("Setup successful! ✅")

# Test basic OpenCV functionality
test_image = np.zeros((100, 100, 3), dtype=np.uint8)
print("Test image shape:", test_image.shape)
print("All libraries working correctly!")
