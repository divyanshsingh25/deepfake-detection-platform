# 🛡 DeepShield
## AI-Based Deepfake Detection and Cybercrime Reporting Assistance System

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

DeepShield is a major college project that uses deep learning to detect deepfake
images and videos. It provides:

- **Real/Fake classification** with confidence score
- **Grad-CAM explainability** heatmaps
- **Downloadable forensic PDF reports**
- **Redirect to India's Cyber Crime Portal** (cybercrime.gov.in)
- **Real-time webcam detection**
- **Model comparison** (ResNet50 vs EfficientNet-B0)

---

## 📁 Folder Structure

```
DeepShield/
│
├── dataset/                    ← Place your dataset here
│   ├── videos/
│   │   ├── real/               ← Real videos (.mp4)
│   │   └── fake/               ← Fake/manipulated videos (.mp4)
│   ├── frames/                 ← Auto-generated extracted frames
│   │   ├── real/
│   │   └── fake/
│   └── split/                  ← Auto-generated 70/15/15 split
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   └── deepfake_model.pth      ← Trained model weights
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        ← Frame extraction, dataset, augmentation
│   ├── face_extraction.py      ← MTCNN face detection and cropping
│   ├── gradcam.py              ← Grad-CAM visualisation
│   ├── report_generator.py     ← PDF forensic report generation
│   └── voting.py               ← Majority voting for video classification
│
├── train.py                    ← Model training pipeline
├── evaluate.py                 ← Evaluation metrics and plots
├── app.py                      ← Streamlit frontend
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/yourname/deepshield
cd DeepShield
pip install -r requirements.txt
```

### 2. Download Dataset

**Option A: FaceForensics++ (recommended)**
```bash
# Request access at: https://github.com/ondyari/FaceForensics
# Download c23 (light compression) version
```

**Option B: Kaggle Deepfake Detection Challenge**
```bash
pip install kaggle
kaggle competitions download -c deepfake-detection-challenge
```

### 3. Prepare Dataset
```python
from utils.preprocessing import extract_frames_from_folder, split_dataset

# Step 1: Extract frames from videos
extract_frames_from_folder("dataset/videos/real", "dataset/frames", "real")
extract_frames_from_folder("dataset/videos/fake", "dataset/frames", "fake")

# Step 2 (optional): Crop faces
from utils.face_extraction import process_image_folder, get_face_detector
detector = get_face_detector()
process_image_folder("dataset/frames", "dataset/faces", detector)

# Step 3: Split dataset 70/15/15
split_dataset("dataset/faces", "dataset/split")
```

### 4. Train the Model
```bash
# Train ResNet50
python train.py --model resnet50 --data_dir dataset/split --epochs 15

# Train EfficientNet-B0
python train.py --model efficientnet --data_dir dataset/split --epochs 15 \
                --save_path models/deepfake_model_efficientnet.pth
```

### 5. Evaluate
```bash
python evaluate.py --model_path models/deepfake_model.pth \
                   --model_name resnet50 \
                   --test_dir dataset/split/test
```

### 6. Launch Frontend
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 🧠 Model Architecture

```
Input Image (224×224×3)
       ↓
   MTCNN Face Detection
       ↓
   ResNet50 / EfficientNet-B0
   (Pretrained ImageNet weights)
       ↓
   Frozen Backbone Layers
       ↓
   Unfrozen Last Block (Fine-tuning)
       ↓
   Dropout(0.5)
       ↓
   FC(2048 → 512) + ReLU
       ↓
   Dropout(0.25)
       ↓
   FC(512 → 2) [Real / Fake]
       ↓
   Softmax → Confidence Score
```

---

## 📊 Evaluation Metrics

After training, DeepShield reports:
- Accuracy
- Precision (Fake class)
- Recall (Fake class)
- F1-Score (Fake class)
- Confusion Matrix
- ROC Curve (AUC)

---

## 🎯 Risk Levels

| Label | Confidence | Risk |
|-------|-----------|------|
| Real  | ≥90%      | Low  |
| Fake  | 60–85%    | Medium |
| Fake  | ≥85%      | High |

---

## 🔥 Grad-CAM Explainability

DeepShield uses Gradient-weighted Class Activation Mapping to highlight
which facial regions contributed most to the "Fake" prediction.
Warm colours (red/orange) = high activation (suspicious regions).

---

## 🎬 Video Analysis Pipeline

```
Video File
    ↓
Extract 20 evenly-spaced frames
    ↓
For each frame:
    → MTCNN face detection
    → Model inference → (real_prob, fake_prob)
    ↓
Ensemble Voting:
    → Hard majority vote
    → Soft average vote
    → Weighted confidence vote
    ↓
Final verdict + risk level + PDF report
```

---

## 🔒 Ethics & Responsible Use

- This tool is for **detection only**, not deepfake creation
- No user data is stored or transmitted
- Results must be confirmed by a forensics expert before legal action
- Misuse for false accusations is a criminal offence

---

## 🚀 Free Deployment

### Hugging Face Spaces (Recommended)
```bash
# Create new Space → SDK: Streamlit
git init
git remote add origin https://huggingface.co/spaces/USERNAME/deepshield
git add . && git commit -m "Initial commit"
git push -u origin main
```

### Render Free Tier
- Push to GitHub
- New Web Service on Render → connect GitHub repo
- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 📖 Viva Q&A (20 Questions)




---

## 👥 Team & Acknowledgements

Built as a Major College Project using only free, open-source tools.
No paid APIs or cloud services required.

**Cyber Crime Helpline (India): 1930**
**Portal: https://cybercrime.gov.in**
