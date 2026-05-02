<div align="center">

# VidShield

### A Joint Framework for Video Deepfake Detection and Frame-Level Provenance Protection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C2PA v1.3](https://img.shields.io/badge/C2PA-v1.3-blueviolet)](https://c2pa.org/)
[![IEEE Paper](https://img.shields.io/badge/Paper-IEEE%20Submission-00629B)](docs/)

*Deepfakes do not simply need to be detected, they need to be governed. VidShield does both.*

</div>

---

## The Problem We Set Out to Solve

Deepfake videos have crossed from novelty to genuine threat. Fabricated statements by public officials, non-consensual synthetic imagery, and AI-forged corporate communications are no longer theoretical concerns they are documented, recurring events. What made this project genuinely challenging was the realization that detection alone is not enough. A deepfake detector can classify a video, but the moment it is wrong (and at scale, it will be wrong), there is no forensic trail, no ownership record, and no way to prove what happened to a frame between creation and distribution.

VidShield was designed around this gap. It is a dual-component framework that approaches synthetic video governance from both ends: detecting forgeries through a domain-adaptive temporal vision transformer, and protecting authentic frames through a three-layer provenance architecture. Neither component was built in isolation from the other together they establish a practical foundation for real-world synthetic media governance.

---

## Results at a Glance

| Component | Metric | Value |
|---|---|---|
| **Detection** | Validation AUC | **0.9982** |
| **Detection** | F1-Score | **0.9451** |
| **Detection** | Normalized Robustness Score (8 corruptions) | **0.9185** |
| **Protection — Layer 1 (UAP)** | Average CLIP Similarity Post-Perturbation | **0.5853** |
| **Protection — Layer 2 (DCT)** | Average Fingerprint Recovery Accuracy | **0.9572** |
| **Protection — Layer 3 (C2PA)** | Tamper Detection | **Protected results in simulated tests** |

---

## How It Works

### Detection

The detection component is built around a **Video Swin Transformer Small (VST-S)** pretrained on Kinetics-400, fine-tuned on a joint corpus from **DFDC** and **DeeperForensics-1.0** under strict identity-disjoint splits. To prevent the model from memorising dataset-specific compression artifacts, a **Gradient Reversal Layer (GRL)** is added as an adversarial branch that forces the backbone to learn domain-invariant forgery cues.

This was not the first model we trained. We started with a **ConvNeXt-tiny CNN spatial baseline**, a disciplined first step that taught us quickly where purely spatial, frame-level features fall short. The shift to a temporal transformer backbone was driven by that failure analysis. The video swin model sees sequences, not snapshots, and that makes all the difference.

### Protection

The protection pipeline applies three sequential, independently-verifiable security layers to each video frame.

- **Layer 1 — Universal Adversarial Perturbation (UAP):** A single input-agnostic noise pattern trained on 2,000 CelebA images to minimize cosine similarity in CLIP's embedding space. Applied once, it disrupts AI comprehension of any subsequent frame without being visible to the human eye.

- **Layer 2 — DCT Frequency-Domain Fingerprinting:** The frame is divided into 8×8 blocks, and a 32-bit SHA-256-derived ownership fingerprint is embedded at DCT coefficient position (4,3) with 12-fold majority-vote redundancy. It survives JPEG compression and UAP noise. Decoding requires a secret seed — making unauthorized extraction cryptographically resistant.

- **Layer 3 — C2PA v1.3 Provenance Manifest:** A versioned, hash-linked JSON manifest is generated for each protected frame, recording the SHA-256 fingerprint hash in a chain of custody. Any unauthorized modification triggers a hash mismatch. Authorized edits are logged as a new version entry in the chain, allowing verifiers to distinguish malicious tampering from legitimate post-production.

The protection design also evolved considerably. Before settling on this architecture, we explored **Fernet symmetric encryption-based watermarking** and validated it through tampering simulation. While cryptographically sound, Fernet-based embedding proved too brittle under lossy post-processing conditions — the shift to DCT-domain embedding was the direct result of that lesson.

---

## Repository Structure

```
VidShield/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── detection/                              # Deepfake detection component
│   │
│   ├── v1_spatial_convnext/               # Phase 1: CNN spatial baseline (ConvNeXt-tiny)
│   │   └── convnext_training.ipynb        #   Full training run with confusion matrix + ROC curve
│   │
│   └── v2_video_swin/                     # Phase 2: Final VST-S + GRL model (production)
│       │
│       ├── training.ipynb                 #   Full training run — loss curves, epoch metrics
│       ├── cross_dataset_testing.ipynb    #   Zero-shot eval on Celeb-DF-v2
│       ├── model_analysis.ipynb           #   Intra-dataset breakdown + hard sample analysis
│       │
│       ├── unified_training_metadata.csv  #   Per-epoch train/val metrics log
│       │
│       └── assets/                        #   All output visualisations
│           ├── training_curves.png                  # Loss + AUC across epochs
│           ├── top_14_failure_samples.png           # Hardest misclassified frames
│           ├── validation_pr_curve.png              # Precision-Recall on val set
│           ├── validation_confusion_matrix.png      # Val confusion matrix
│           ├── validation_auc.png                   # Val ROC-AUC curve
│           ├── cross_pr_curve.png                   # PR curve on Celeb-DF-v2
│           ├── cross_confusion_matrix.png           # Celeb-DF-v2 confusion matrix
│           └── gradcam/                             # GradCAM saliency maps (5 images)
│               ├── deeperforensics_gradcam_01.png   #   DeeperForensics-1.0 sample 1
│               ├── deeperforensics_gradcam_02.png   #   DeeperForensics-1.0 sample 2
│               ├── dfdc_gradcam_01.png              #   DFDC sample 1
│               ├── dfdc_gradcam_02.png              #   DFDC sample 2
│               └── dfdc_gradcam_03.png              #   DFDC sample 3
│
└── protection/                             # Frame-level provenance protection component
    │
    ├── v1_fernet_watermark/               # Phase 1: Fernet encryption-based watermarking
    │   └── flask_app/                     #   Flask web app for embed + tamper simulation
    │       ├── app.py
    │       ├── templates/
    │       └── static/
    │
    └── v2_uap_dct_c2pa/                   # Phase 2: Final three-layer pipeline (production)
        │
        ├── uap_trainer_clip_eot_finetune.py   # UAP training — CLIP loss + EoT fine-tuning
        ├── apply_uap.py                        # Apply trained UAP pattern to input frames
        ├── data_loader.py                      # CelebA + personal frame loading utilities
        │
        ├── dct_fingerprint_redundant_v2.py    # DCT embed core — 12× redundancy, majority vote
        ├── embed_fingerprint.py                # Fingerprint embedding entry point
        ├── extract_fingerprint.py              # Fingerprint extraction entry point
        ├── extract_fingerprint_test.py         # Recovery accuracy test harness
        │
        ├── create_provenance.py                # Generate C2PA v1.3 manifest (version 0)
        ├── update_chain.py                     # Append new version entry to manifest chain
        ├── verify_chain.py                     # Full chain integrity verification
        ├── verify_provenance.py                # Single-frame provenance check
        │
        ├── clip_evaluation.py                  # CLIP similarity scoring (pre vs. post UAP)
        ├── caption_test.py                     # BLIP caption drift analysis
        ├── requirements.txt                    # Python dependencies for this module
        │
        └── assets/                             # Evaluation outputs and result tables
            ├── c2pa_chaining_sample_output.png     # Sample C2PA manifest chain output
            ├── blip_caption_comparison_table.png   # Original vs. UAP-protected captions
            └── clip_dct_results_table.png          # Per-image CLIP scores + DCT recovery accuracy
```

### A note on the versioned folder pattern

Each component has a `v1_` and `v2_` folder. This is intentional. Research projects rarely arrive at their final architecture in one shot, and erasing the path taken loses information that future contributors (or your future self) will find genuinely useful. The `v1_` folders preserve the first-generation approaches — the ConvNeXt spatial baseline and the Flask-backed Fernet watermarking prototype — not as failed experiments to be embarrassed about, but as documented decisions with documented reasons for moving on.

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (experiments run on NVIDIA RTX 6000 Ada; any CUDA-enabled GPU with ≥16GB VRAM recommended)
- PyTorch 2.7

### Installation

```bash
git clone https://github.com/adityar9764/VidShield.git
cd VidShield

# Create a conda environment (recommended)
conda env create -f environment.yml
conda activate vidshield

# Or install via pip
pip install -r requirements.txt
```

### Dataset Setup

VidShield was trained and evaluated on:

| Dataset | Role | Access |
|---|---|---|
| [DFDC](https://www.kaggle.com/c/deepfake-detection-challenge/data) | Detection training (in-the-wild fakes) | Kaggle |
| [DeeperForensics-1.0](https://github.com/EndlessSora/DeeperForensics-1.0) | Detection training (high-fidelity fakes) | GitHub |
| [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | UAP training + protection evaluation | MMLAB |
| [Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics) | Cross-dataset generalization | GitHub |

After downloading, set the dataset root paths in the first cell of the relevant training notebook before running.

---

## Pre-trained Model Weights

Training from scratch requires significant GPU resources and time. To make the work directly reproducible, both model checkpoints are hosted on Google Drive and available for download.

| Model | Phase | Val AUC | Size | Download |
|---|---|---|---|---|
| ConvNeXt-tiny (spatial baseline) | Phase 1 | — | ~110 MB | [Google Drive](https://drive.google.com/drive/folders/1RzefPj7VP_SmDB7o1gRK4urGx83PYH8v?usp=drive_link) |
| Video Swin Transformer Small + GRL | Phase 2 | 0.9982 | ~210 MB | [Google Drive](https://drive.google.com/drive/folders/1o0o6pCLrRLUD7Is9Kr7iUfmYeCzqP-hs?usp=sharing) |


### Using a pre-trained checkpoint

Download the `.pth` file and place it anywhere convenient, then point the relevant notebook to it via the checkpoint path variable in the first cell.

For the protection pipeline's UAP pattern (also a trained artifact), the pre-trained perturbation file is included in the same Drive folder.


## Training the Detector

All detection experiments are notebook-driven. Open the relevant notebook and run cells top-to-bottom dataset paths are configured in the first cell.

```
# Phase 2: Video Swin Transformer + GRL (recommended)
detection/v2_video_swin/training.ipynb

# Phase 1: ConvNeXt-tiny spatial baseline (for reference)
detection/v1_spatial_convnext/convnext_training.ipynb
```

The training notebook logs per-epoch metrics to `unified_training_metadata.csv` and saves visualisations (training curves, validation confusion matrix, PR curve, AUC plot) to `v2_video_swin/assets/`.

**Key training decisions documented in the notebook:**

- AdamW optimizer, fixed lr = 3×10⁻⁵, 10 epochs
- Domain-balanced batches: 5 real DFDC + 5 fake DFDC + 5 real DF1.0 + 5 fake DF1.0 per batch
- GRL lambda scheduled: warm-up for 2 epochs, linear ramp to λmax = 0.05 over epochs 2–5, constant thereafter
- Label smoothing ε = 0.05 on binary cross-entropy
- Augmentation emphasis: JPEG compression (p = 0.60) and resolution downscaling (p = 0.70)

---

## Running the Protection Pipeline

Install dependencies for the protection module first:

```bash
pip install -r protection/v2_uap_dct_c2pa/requirements.txt
```

**Step 1 — Train the UAP** (or skip if you have a pre-trained perturbation):

```bash
python protection/v2_uap_dct_c2pa/uap_trainer_clip_eot_finetune.py \
    --data path/to/celeba/ \
    --output uap_pattern.pth
```

**Step 2 — Apply UAP to frames:**

```bash
python protection/v2_uap_dct_c2pa/apply_uap.py \
    --input path/to/frames/ \
    --uap uap_pattern.pth \
    --output path/to/uap_protected/
```

**Step 3 — Embed DCT fingerprint:**

```bash
python protection/v2_uap_dct_c2pa/embed_fingerprint.py \
    --input path/to/uap_protected/ \
    --seed 2026 \
    --output path/to/fingerprinted/
```

**Step 4 — Generate C2PA provenance manifest:**

```bash
python protection/v2_uap_dct_c2pa/create_provenance.py \
    --frames path/to/fingerprinted/ \
    --output path/to/manifests/
```

**To verify a frame's provenance:**

```bash
python protection/v2_uap_dct_c2pa/verify_chain.py \
    --frame path/to/protected_frame.jpg \
    --manifest path/to/manifest.json
```

A clean match at version 0 confirms the frame is unmodified. A match at a later version confirms a recorded, authorized modification. A complete mismatch at all versions is a tamper flag.

**To log an authorized modification to the chain:**

```bash
python protection/v2_uap_dct_c2pa/update_chain.py \
    --frame path/to/modified_frame.jpg \
    --manifest path/to/manifest.json
```

---

## Evaluation

**Detection — model analysis and cross-dataset generalization:**

```
# Intra-dataset breakdown + hard sample / failure analysis
detection/v2_video_swin/model_analysis.ipynb

# Zero-shot generalization on Celeb-DF-v2
detection/v2_video_swin/cross_dataset_testing.ipynb
```

GradCAM saliency maps for the 5 most informative failure cases are pre-generated in `detection/v2_video_swin/assets/gradcam/`.

**Protection — CLIP, BLIP, and provenance evaluation:**

```bash
# CLIP similarity scoring (pre vs. post UAP)
python protection/v2_uap_dct_c2pa/clip_evaluation.py \
    --frames path/to/uap_protected/ --original path/to/original/

# BLIP caption drift analysis
python protection/v2_uap_dct_c2pa/caption_test.py \
    --frames path/to/uap_protected/ --original path/to/original/

# DCT fingerprint recovery accuracy
python protection/v2_uap_dct_c2pa/extract_fingerprint_test.py \
    --frames path/to/fingerprinted/ --seed 2026

# Provenance chain integrity verification
python protection/v2_uap_dct_c2pa/verify_chain.py \
    --frames path/to/protected/ --manifests path/to/manifests/
```

Pre-computed result tables are saved in `protection/v2_uap_dct_c2pa/assets/` for reference without re-running.

### Robustness Benchmark

The NRS evaluation suite covers 8 corruption scenarios:

| Scenario | AUC | Accuracy |
|---|---|---|
| Baseline (clean) | 0.9982 | 0.9590 |
| Motion Blur | 0.9946 | 0.9407 |
| JPEG QF 50 | 0.9982 | 0.9331 |
| JPEG QF 30 | 0.9982 | 0.9331 |
| ISO Noise | 0.9951 | 0.9510 |
| Downscale ×0.25 | 0.9982 | 0.9334 |
| Hue/Saturation Shift | 0.9939 | 0.9353 |
| CoarseDropout (small) | 0.9982 | 0.9378 |
| CoarseDropout (large) | 0.9950 | 0.9396 |

> **Note on salt-and-pepper noise:** A separate test at 5% impulse noise produced AUC = 0.9381, revealing a qualitatively different failure mode from smooth corruptions. This is a documented limitation, not an omission.

---

## Known Limitations

We believe reproducibility includes being honest about where a system falls short.

**On detection:** The cross-dataset evaluation on Celeb-DF-v2 returns AUC = 0.5415 which is effectively near-random. The model predominantly predicts "fake" on nearly all samples, reflecting the qualitative gap between GAN-based synthesis in the training corpora and the higher-fidelity blending techniques in Celeb-DF-v2. Multi-source domain training is necessary but not sufficient for generalisation across qualitatively different synthesis pipelines.

**On protection:** CLIP similarity disruption is not uniform across images. High-contrast subjects on uniform backgrounds (e.g., image `000004` at 0.7921) show higher residual similarity, suggesting that content-adaptive UAP generation would be a meaningful direction for future work. One of 19 evaluation images produced identical BLIP captions before and after UAP application, indicating that the current perturbation budget may be insufficient for certain image types.

---

## Architecture Deep Dives

### Detection: Why Video Swin over a pure CNN?

The ConvNeXt-tiny baseline (Phase 1) validated that spatial artifact detection looking at individual frames has a ceiling. Face-swap and reenactment deepfakes increasingly produce per-frame content that is visually clean. The tell is in motion: temporal discontinuities, unnatural blink patterns, micro-expression artifacts that manifest across frame boundaries. A temporal vision transformer with shifted window attention is architecturally suited to catching exactly these cross-frame anomalies.

### Protection: Why DCT over Fernet?

Fernet is a symmetric encryption scheme which is robust and cryptographically sound when you control both ends of the channel. The problem is survivability: a Fernet watermark embedded in raw pixel values does not survive JPEG re-encoding, which is how nearly all online video is re-distributed. DCT-domain embedding targets mid-frequency coefficients that JPEG compression specifically preserves (it selectively discards high-frequency components). The 12-fold repetition with majority-vote decoding is the engineering solution that makes frequency-domain embedding robust in practice, not just in theory.

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{vidshield2026,
  title     = {A Joint Framework for Video Deepfake Detection and Frame-Level Provenance Protection},
  author    = {Aditya Raj, Sarthak Harade, Jayan Karkera,Dr. Manoj Sabnis, Mohit Ailani},
  conference   = {IEEE},
  year      = {2026},
  note      = {Department of Information Technology, Vivekanand Education Society's Institute of Technology, Mumbai}
}
```

---

## Acknowledgements

This project was developed under the guidance of **Dr. Manoj Sabnis**, Department of Information Technology, VESIT Mumbai, whose feedback and support across every iteration of this work from the ConvNeXt baseline through the final VidShield architecture was invaluable.

The following open-source datasets were used in this work: DFDC (Meta AI), DeeperForensics-1.0, CelebA (MMLAB), and Celeb-DF-v2. We thank their creators for making these resources publicly available.

---


<div align="center">
<sub>Built at VESIT Mumbai · Department of Information Technology · 2024–2025</sub>
</div>
