import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = "./test_images"      # images to protect
OUTPUT_DIR = "./protected_images"
UAP_PATH = "./uap_noise_clip_eot_refined.pt"
IMG_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load UAP
uap = torch.load(UAP_PATH, map_location=DEVICE)
uap = uap.to(DEVICE)

# Image transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

# Apply UAP
for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)

    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    protected = torch.clamp(image_tensor + uap, 0, 1)

    protected_img = to_pil(protected.squeeze().cpu())
    protected_img.save(os.path.join(OUTPUT_DIR, img_name))

print("✅ UAP applied and protected images saved.")
