import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os
import numpy as np
import cv2

# Public, ungated model
MODEL_ID = "umm-maybe/AI-image-detector"

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

def predict_pretrained(image_path):
    """
    Returns fake probability between 0 and 1
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]

        # Labels: [REAL, FAKE]
        fake_prob = float(probs[1])
        return fake_prob

    except Exception as e:
        print("Pretrained model error:", e)
        return None


def predict_pretrained_video(face_folder, max_faces=10):
    """
    Runs pretrained model on sampled face frames
    Returns fake probability (0–1)
    """
    fake_scores = []

    face_files = sorted(os.listdir(face_folder))[:max_faces]

    for face in face_files:
        face_path = os.path.join(face_folder, face)
        prob = predict_pretrained(face_path)
        if prob is not None:
            fake_scores.append(prob)

    if not fake_scores:
        return None

    return float(np.median(fake_scores))