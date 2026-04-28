from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)

model.eval()

def caption_image(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=40)
    return processor.decode(out[0], skip_special_tokens=True)

# Base paths
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ORIG_DIR = os.path.join(BASE_DIR, "test_images")
# PROT_DIR = os.path.join(BASE_DIR, "final_protected", "class_dummy")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ORIG_DIR = os.path.join(BASE_DIR, "test_images")
PROT_DIR = os.path.join(BASE_DIR, "final_protected", "class_dummy")

# Allowed image formats
VALID_EXT = (".jpg", ".jpeg", ".png")

for img in os.listdir(ORIG_DIR):

    if not img.lower().endswith(VALID_EXT):
        continue

    orig_path = os.path.join(ORIG_DIR, img)
    prot_path = os.path.join(PROT_DIR, img)

    if not os.path.exists(prot_path):
        print(f"\nIMAGE: {img}")
        print("Protected version missing — skipped")
        continue

    print("\nIMAGE:", img)
    print("Original caption  :", caption_image(orig_path))
    print("Protected caption :", caption_image(prot_path))
