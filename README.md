# 🛡️ DeepShield — AI Deepfake Detection System

<div align="center">

![DeepShield](https://img.shields.io/badge/DeepShield-v1.0-2563eb?style=for-the-badge&logoColor=white)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

> **AI-powered deepfake detection with Grad-CAM explainability, forensic PDF reports, and India Cybercrime Portal integration.**

</div>

---

## ✨ Features

- 🔍 **Real / Fake classification** — ResNet50 & EfficientNet-B0 ensemble
- 🔥 **Grad-CAM heatmaps** — visual explanation of model decisions
- 🎥 **Image & Video support** — frame-by-frame video analysis
- 📊 **Confidence scoring** — risk level (LOW / MEDIUM / HIGH)
- 📄 **Forensic PDF reports** — downloadable evidence documents
- 📷 **Live webcam detection** — real-time face analysis
- 🌐 **Cybercrime portal** — direct link to cybercrime.gov.in
- 🔒 **Privacy first** — 100% local processing, no cloud uploads

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| Python | 3.9 | 3.10 or 3.11 |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB free | 20 GB free |
| GPU | Not required | NVIDIA CUDA (faster inference) |

---

## 🚀 Installation Guide

### Step 1 — Clone the Repository

```bash
git clone https://github.com/divyanshsingh25/deepfake-detection-platform.git
cd deepfake-detection-platform
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install PyTorch (CPU)

```bash
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu --timeout 300
```

> If you have an NVIDIA GPU, install the CUDA version instead:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### Step 4 — Install Remaining Dependencies

```bash
pip install streamlit opencv-python mtcnn facenet-pytorch reportlab scikit-learn matplotlib numpy Pillow tqdm albumentations seaborn --timeout 120
```

### Step 5 — Add Your Trained Model

Place your trained model file inside the `models/` folder:

```
models/
└── deepfake_model.pth        ← required for ResNet50
└── deepfake_model_efficientnet.pth  ← optional for EfficientNet
```

> **Don't have a model yet?** Train one — see the Training section below.

### Step 6 — Launch the App

```bash
# Windows (use this if plain "streamlit" doesn't work)
.venv\Scripts\python -m streamlit run app.py

# Linux / macOS
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🧠 Training Your Own Model

### Prepare Your Dataset

Your dataset folder should look like this:

```
dataset/
├── split/
│   ├── train/
│   │   ├── real/    ← real face images
│   │   └── fake/    ← deepfake face images
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
```

**Recommended datasets:**
- [FaceForensics++](https://github.com/ondyari/FaceForensics) — request access
- [Kaggle DFDC](https://www.kaggle.com/c/deepfake-detection-challenge) — open download

### Extract Frames from Videos (optional)

```python
from utils.preprocessing import extract_frames_from_folder, split_dataset

extract_frames_from_folder("dataset/videos/real", "dataset/frames", "real")
extract_frames_from_folder("dataset/videos/fake", "dataset/frames", "fake")
split_dataset("dataset/frames", "dataset/split")
```

### Train

```bash
# Train ResNet50 (default)
python train.py --model resnet50 --data_dir dataset/split --epochs 15

# Train EfficientNet-B0
python train.py --model efficientnet --data_dir dataset/split --epochs 15 --save_path models/deepfake_model_efficientnet.pth
```

### Evaluate

```bash
python evaluate.py --model_path models/deepfake_model.pth --model_name resnet50 --test_dir dataset/split/test
```

This generates: confusion matrix, ROC curve, accuracy, F1-score — all saved to `models/`.

---

## 📁 Project Structure

```
deepfake-detection-platform/
├── app.py                      ← Streamlit frontend
├── train.py                    ← Model training pipeline
├── evaluate.py                 ← Evaluation & plots
├── requirements.txt
├── packages.txt                ← System packages (for cloud deploy)
├── .streamlit/
│   └── config.toml             ← Streamlit theme config
├── models/
│   ├── deepfake_model.pth      ← Trained ResNet50 weights
│   ├── deepfake_model_plot.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── utils/
    ├── __init__.py
    ├── face_extraction.py      ← MTCNN + Haar Cascade face detector
    ├── gradcam.py              ← Grad-CAM implementation
    ├── preprocessing.py        ← Transforms & dataset loader
    ├── report_generator.py     ← PDF forensic report
    └── voting.py               ← Ensemble voting logic
```

---

## 🧠 Model Architecture

```
Input Image (224×224×3)
        ↓
   MTCNN Face Detection
        ↓
   ResNet50 / EfficientNet-B0
   (ImageNet pretrained backbone)
        ↓
   Dropout(0.5)
        ↓
   FC(2048 → 512) + ReLU
        ↓
   Dropout(0.25)
        ↓
   FC(512 → 2)  →  Softmax
        ↓
   Real / Fake  +  Confidence %
```

---

## 🎯 Risk Levels

| Verdict | Confidence | Risk Level |
|---------|-----------|------------|
| Real | Any | LOW |
| Fake | 60 – 84% | MEDIUM |
| Fake | ≥ 85% | HIGH |

---

## 🔥 Grad-CAM Explainability

Grad-CAM highlights which face regions the model focused on. Warm colours (red/orange) indicate high activation — the areas most responsible for the Fake prediction.

---

## 🎬 Video Analysis Pipeline

```
Video File → Extract 20 evenly-spaced frames
                    ↓
        For each frame: MTCNN face crop → Model inference
                    ↓
        Ensemble voting (hard + soft + weighted)
                    ↓
        Final verdict + Risk level + PDF report
```

---



---

## ⚖️ Ethics & Responsible Use

- This tool is for **detection only** — not deepfake creation
- Results are probabilistic — verify with a forensics expert before legal action
- No facial data is retained after the session ends
- Compliant with **DPDP Act 2023** (India) privacy principles
- Misuse to falsely accuse individuals is a criminal offence

---

## 🚨 Report Cybercrime

If you encounter a deepfake being used for fraud, harassment, or blackmail:

**🌐 Portal:** https://cybercrime.gov.in  
**📞 Helpline:** 1930 (India National Cyber Crime Helpline)

---

## 📚 References

1. Rössler et al. — *FaceForensics++*, ICCV 2019
2. Selvaraju et al. — *Grad-CAM*, ICCV 2017
3. He et al. — *Deep Residual Learning for Image Recognition*, CVPR 2016
4. Tan & Le — *EfficientNet*, ICML 2019
5. Zhang et al. — *MTCNN*, IEEE Signal Processing Letters 2016
6. Dolhansky et al. — *Deepfake Detection Challenge*, NeurIPS 2020

---
