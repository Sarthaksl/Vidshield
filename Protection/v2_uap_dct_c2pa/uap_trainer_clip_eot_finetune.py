import os
import random
import torch
import clip
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

# =====================
# CONFIG
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "./celebA/img_align_celeba"
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_IMAGES = 2000

EPSILON = 10 / 255          # keep same epsilon
LR = 0.005                  # LOWER LR for fine-tuning
UAP_STEPS = 4               # few steps only
EOT_SAMPLES = 3

UAP_INIT_PATH = "./uap_noise_clip.pt"
UAP_SAVE_PATH = "./uap_noise_clip_eot_refined.pt"

# =====================
# DATASET
# =====================
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

# =====================
# BASE TRANSFORM
# =====================
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

dataset = CelebAImages(DATA_PATH, base_transform, NUM_IMAGES)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =====================
# LOAD CLIP
# =====================
model, _ = clip.load("ViT-B/32", device=DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# =====================
# EOT TRANSFORM
# =====================
def random_eot(x):
    scale = random.uniform(0.9, 1.0)
    size = int(IMG_SIZE * scale)
    x = TF.resize(x, size)
    x = TF.center_crop(x, IMG_SIZE)

    if random.random() < 0.5:
        x = TF.gaussian_blur(x, kernel_size=3)

    return x

# =====================
# LOAD EXISTING STRONG UAP
# =====================
uap = torch.load(UAP_INIT_PATH, map_location=DEVICE)
uap.requires_grad_(True)

print("✅ Loaded strong CLIP-UAP for EOT fine-tuning")

# =====================
# EOT FINE-TUNING
# =====================
for step in range(UAP_STEPS):
    print(f"🔁 EOT Fine-Tune Pass {step+1}/{UAP_STEPS}")

    for images in loader:
        images = images.to(DEVICE)

        with torch.no_grad():
            emb_orig = model.encode_image(images)
            emb_orig = emb_orig / emb_orig.norm(dim=-1, keepdim=True)

        total_loss = 0.0

        for _ in range(EOT_SAMPLES):
            perturbed = torch.clamp(images + uap, 0, 1)
            perturbed = random_eot(perturbed)

            emb_adv = model.encode_image(perturbed)
            emb_adv = emb_adv / emb_adv.norm(dim=-1, keepdim=True)

            total_loss += -F.cosine_similarity(emb_orig, emb_adv).mean()

        loss = total_loss / EOT_SAMPLES
        loss.backward()

        with torch.no_grad():
            uap += LR * uap.grad.sign()
            uap.clamp_(-EPSILON, EPSILON)

        uap.grad.zero_()

    print("   ✓ Pass complete")

# =====================
# SAVE
# =====================
torch.save(uap.detach().cpu(), UAP_SAVE_PATH)
print(f"✅ Refined EOT-robust UAP saved as {UAP_SAVE_PATH}")
