import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# CONFIG
DATA_PATH = "./celebA/img_align_celeba"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_IMAGES = 2000

class CelebAImages(Dataset):
    def __init__(self, image_dir, transform, num_images):
        self.image_dir = image_dir
        self.transform = transform
        self.images = random.sample(os.listdir(image_dir), num_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)

def get_celeba_loader():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    dataset = CelebAImages(DATA_PATH, transform, NUM_IMAGES)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    print(f"✅ Loaded {len(dataset)} CelebA images (image-only)")
    return loader
