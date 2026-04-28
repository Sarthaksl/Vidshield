import os
import torch
import clip
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ORIG_DIR = "./test_images"
PROT_DIR = "./protected_images"
IMG_SIZE = 224

# Load CLIP
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

def get_embedding(img_path):
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb

scores = []

for img_name in os.listdir(ORIG_DIR):
    orig_path = os.path.join(ORIG_DIR, img_name)
    prot_path = os.path.join(PROT_DIR, img_name)

    if not os.path.exists(prot_path):
        continue

    emb_orig = get_embedding(orig_path)
    emb_prot = get_embedding(prot_path)

    sim = F.cosine_similarity(emb_orig, emb_prot).item()
    scores.append(sim)

    print(f"{img_name}: CLIP similarity = {sim:.4f}")

print("\n📉 Average CLIP similarity:", sum(scores) / len(scores))
