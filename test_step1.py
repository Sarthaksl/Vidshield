import os, math, yaml, time, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, accuracy_score
import pandas as pd
from tqdm import tqdm
import timm
from einops import rearrange
import cv2
import glob
from rich import print as rprint

ROOT = Path.home()/ "deepfake-detector"
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Data -----------------
class ClipDataset(Dataset):
    def _init_(self, manifest_csv, crops_dir, split, frames_per_clip=32, frame_stride_options=(1,2,3), image_size=224, augment=True):
        self.df = pd.read_csv(manifest_csv)
        self.crops_dir = Path(os.path.expanduser(crops_dir))
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.frame_stride_options = frame_stride_options
        self.image_size = image_size
        self.augment = augment and (split=="train")

        # build a list of clip folders (face-cropped jpgs)
        self.items = []
        for _, row in self.df.iterrows():
            label = "fake" if row["label"]==1 else "real"
            vid_name = Path(row["path"]).stem
            clip_dir = self.crops_dir / split / label / vid_name
            if clip_dir.exists():
                frames = sorted(glob.glob(str(clip_dir/"*.jpg")))
                if len(frames)>=self.frames_per_clip:
                    self.items.append((clip_dir, row["label"]))
        rprint(f"[cyan]{split}[/cyan]: usable videos={len(self.items)}")

        # transforms
        self.train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.RandomResizedCrop(self.image_size, scale=(0.9,1.0)),
            transforms.ColorJitter(0.1,0.1,0.1,0.05),
        ])
        self.eval_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
        ])

    def _sample_indices(self, num_frames):
        stride = random.choice(self.frame_stride_options)
        max_start = max(0, num_frames - self.frames_per_clip*stride)
        start = random.randint(0, max_start) if self.split=="train" else max_start//2
        idxs = [start + i*stride for i in range(self.frames_per_clip)]
        return idxs

    def _load_clip(self, clip_dir):
        frames = sorted(glob.glob(str(clip_dir/"*.jpg")))
        idxs = self._sample_indices(len(frames))
        imgs = []
        for i in idxs:
            i = min(i, len(frames)-1)
            img = cv2.imread(frames[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.augment:
                img = self.train_tf(img)
            else:
                img = self.eval_tf(img)
            imgs.append(img)
        clip = torch.stack(imgs, dim=0)  # [T,3,H,W]
        # random JPEG compression artifact (light)
        if self.augment and random.random()<0.2:
            noise = torch.randn_like(clip)*0.01
            clip = (clip+noise).clamp(0,1)
        return clip

    def _fft_maps(self, clip):
        # clip: [T,3,H,W] in [0,1]
        # convert to gray, take 2D FFT magnitude (log scaled), then stack
        with torch.no_grad():
            gray = 0.2989*clip[:,0]+0.5870*clip[:,1]+0.1140*clip[:,2]  # [T,H,W]
            fft = torch.fft.fft2(gray)
            fft = torch.fft.fftshift(fft)
            mag = torch.log(torch.abs(fft)+1e-6)  # [T,H,W]
            # take a few frames (every 2nd) and stack as channels
            sel = mag[::2]
            # normalize per map
            sel = (sel - sel.mean(dim=(1,2), keepdim=True)) / (sel.std(dim=(1,2), keepdim=True)+1e-6)
            # pick first 8 maps or pad
            if sel.shape[0] >= 8:
                sel = sel[:8]
            else:
                pad = torch.zeros((8-sel.shape[0], sel.shape[1], sel.shape[2]), dtype=sel.dtype, device=sel.device)
                sel = torch.cat([sel, pad], dim=0)
            return sel  # [8,H,W]

    def _len_(self): return len(self.items)

    def _getitem_(self, idx):
        clip_dir, label = self.items[idx]
        clip = self._load_clip(clip_dir)               # [T,3,H,W]
        fft_maps = self._fft_maps(clip).unsqueeze(1)   # [8,1,H,W]
        # normalize RGB like ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        clip = (clip - mean)/std
        return {"clip": clip, "fft": fft_maps, "label": torch.tensor(label, dtype=torch.float32)}

# ----------------- Model -----------------
class VideoBackbone(nn.Module):
    # SlowFast-R50 from pytorchvideo via timm fallback -> use pytorchvideo hub weights through torch.hub if available
    def _init_(self, embed_dim=512):
        super()._init_()
        # use a 3D CNN from pytorchvideo (x3d_m is also good). To keep dependencies simple, approximate with (Timm 2D + temporal pooling).
        # For reliability here, we’ll implement a 2D backbone (ConvNeXt-Tiny) applied per-frame + temporal attention pooling.
        self.frame_enc = timm.create_model("convnext_tiny", pretrained=True, num_classes=0, global_pool="avg")
        self.proj = nn.Linear(self.frame_enc.num_features, embed_dim)
        self.temporal_att = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

    def forward(self, clip):  # [B,T,3,224,224]
        B,T,_,H,W = clip.shape
        x = clip.view(B*T,3,H,W)
        feat = self.frame_enc(x)           # [B*T, C]
        feat = self.proj(feat)             # [B*T, E]
        feat = feat.view(B,T,-1)           # [B,T,E]
        # temporal attention pooling
        attn_out, _ = self.temporal_att(feat, feat, feat)
        vid_emb = attn_out.mean(dim=1)     # [B,E]
        return vid_emb

class FreqBackbone(nn.Module):
    def _init_(self, embed_dim=256):
        super()._init_()
        # EfficientNetV2-S expects 3-ch; we have 8×1 maps. Pack as 3-ch by grouping or project with a small conv.
        self.reduce = nn.Conv2d(8, 3, kernel_size=1)
        self.enc = timm.create_model("efficientnetv2_s", pretrained=True, num_classes=0, global_pool="avg")
        self.proj = nn.Linear(self.enc.num_features, embed_dim)

    def forward(self, fft_maps):  # [B,8,1,H,W] or [B,8,H,W]
        if fft_maps.dim()==5:
            fft_maps = fft_maps.squeeze(2)   # [B,8,H,W]
        x = self.reduce(fft_maps)            # [B,3,H,W]
        feat = self.enc(x)                   # [B,C]
        return self.proj(feat)               # [B,E]

class TwoStreamFusion(nn.Module):
    def _init_(self, fusion_dim=768, dropout=0.2):
        super()._init_()
        self.video = VideoBackbone(embed_dim=fusion_dim//2)
        self.freq  = FreqBackbone(embed_dim=fusion_dim//2)
        self.bn = nn.BatchNorm1d(fusion_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim//2, 1)
        )

    def forward(self, clip, fft_maps):
        v = self.video(clip)     # [B,f/2]
        f = self.freq(fft_maps)  # [B,f/2]
        x = torch.cat([v,f], dim=1)
        x = self.bn(x)
        x = self.drop(x)
        logits = self.head(x).squeeze(1)
        return logits

# ----------------- Utils -----------------
def bce_focal_loss(logits, targets, label_smooth=0.05, gamma=1.5):
    # mix BCE + Focal 50/50
    targets = targets.clamp(0,1)
    targets = targets*(1-label_smooth) + 0.5*label_smooth
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = torch.where(targets>0.5, p, 1-p)
    focal = (1-pt).pow(gamma) * bce
    return 0.5*bce.mean() + 0.5*focal.mean()

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        clip = batch["clip"].to(device)         # [B,T,3,H,W]
        fftm = batch["fft"].to(device)          # [B,8,1,H,W]
        y = batch["label"].to(device)
        logits = model(clip, fftm)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits).numpy()
    y_true = labels.numpy().astype(int)

    # Metrics (threshold chosen later)
    auc = roc_auc_score(y_true, probs)
    ap = average_precision_score(y_true, probs)
    return probs, y_true, {"roc_auc": float(auc), "pr_auc": float(ap)}

def pick_threshold_eer(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(thr[idx])

def metrics_at_threshold(y_true, probs, thr):
    y_pred = (probs >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0,1])
    return {"accuracy": float(acc), "f1": float(f1), "confusion_matrix": cm.tolist(), "threshold": float(thr)}

def make_loader(manifest, cfg, split):
    ds = ClipDataset(
        manifest_csv=manifest,
        crops_dir=cfg["data"]["crops_dir"],
        split=split,
        frames_per_clip=cfg["data"]["frames_per_clip"],
        frame_stride_options=tuple(cfg["data"]["frame_stride_options"]),
        image_size=cfg["data"]["image_size"],
        augment=True
    )
    bs = cfg["train"]["batch_size"]
    num_workers = cfg["train"]["num_workers"]
    return DataLoader(ds, batch_size=bs, shuffle=(split=="train"), num_workers=num_workers, pin_memory=True, drop_last=(split=="train"))

def main():
    # Load config
    with open(os.path.expanduser("~/deepfake-detector/configs/base.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["train"]["seed"])
    random.seed(cfg["train"]["seed"])
    np.random.seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoStreamFusion(fusion_dim=cfg["model"]["fusion_dim"], dropout=cfg["model"]["dropout"]).to(device)

    train_loader = make_loader(cfg["data"]["train_manifest"], cfg, "train")
    val_loader   = make_loader(cfg["data"]["val_manifest"],   cfg, "val")
    test_loader  = make_loader(cfg["data"]["test_manifest"],  cfg, "test")

    # Param groups: fusion head a bit higher lr
    fusion_params = list(model.head.parameters()) + list(model.bn.parameters())
    main_params = [p for n,p in model.named_parameters() if (p.requires_grad and (p not in fusion_params))]
    optimizer = torch.optim.AdamW(
        [{"params": main_params, "lr": cfg["train"]["lr_main"]},
         {"params": fusion_params, "lr": cfg["train"]["lr_fusion"]}],
        weight_decay=cfg["train"]["weight_decay"]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])
    best_val_auc = -1
    best_paths = []

    # simple cosine schedule
    total_epochs = cfg["train"]["epochs"]
    warmup = cfg["train"]["warmup_epochs"]

    def lr_mult(epoch):
        if epoch < warmup:
            return (epoch+1)/max(1,warmup)
        progress = (epoch - warmup) / max(1,(total_epochs - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    for epoch in range(total_epochs):
        for g in optimizer.param_groups:
            base_lr = cfg["train"]["lr_fusion"] if g["lr"]==cfg["train"]["lr_fusion"] else cfg["train"]["lr_main"]
            g["lr"] = base_lr * lr_mult(epoch)

        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        steps = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch+1}/{total_epochs}")):
            clip = batch["clip"].to(device)
            fftm = batch["fft"].to(device)
            y = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"]):
                logits = model(clip, fftm)
                loss = bce_focal_loss(logits, y, cfg["train"]["label_smoothing"], cfg["train"]["focal_gamma"])

            scaler.scale(loss).backward()
            steps += 1
            if steps % cfg["train"]["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()

        # ----- validation -----
        val_probs, val_true, val_stats = eval_epoch(model, val_loader, device)
        val_auc = val_stats["roc_auc"]
        rprint({"epoch": epoch+1, "train_loss": epoch_loss/max(1,steps), **val_stats})

        # save top-k by val AUC
        ckpt_path = CKPT_DIR / f"model_epoch{epoch+1:03d}_valAUC{val_auc:.4f}.pt"
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch+1}, ckpt_path)
        best_paths.append((val_auc, str(ckpt_path)))
        best_paths = sorted(best_paths, key=lambda x: -x[0])[:cfg["train"]["save_top_k"]]

    # --------- Evaluate the best checkpoint on TEST ---------
    best_ckpt = best_paths[0][1]
    state = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    rprint(f"[green]Loaded best checkpoint:[/green] {best_ckpt}")

    # choose threshold on VAL by EER
    val_probs, val_true, _ = eval_epoch(model, val_loader, device)
    thr = pick_threshold_eer(val_true, val_probs)

    test_probs, test_true, test_stats = eval_epoch(model, test_loader, device)
    test_report = metrics_at_threshold(test_true, test_probs, thr)
    full_report = {"val": {"threshold_eer": thr}, "test": {**test_stats, **test_report}}

    out_json = LOG_DIR / "final_metrics.json"
    with open(out_json, "w") as f:
        json.dump(full_report, f, indent=2)
    rprint("[bold yellow]Final metrics saved at:[/bold yellow]", str(out_json))

    # also save a clean final model
    final_path = CKPT_DIR / "deepfake_two_stream_final.pth"
    torch.save({"model": model.state_dict(), "cfg": cfg}, final_path)
    rprint("[bold green]Saved final model:[/bold green]", str(final_path))

if _name_ == "_main_":
    main()