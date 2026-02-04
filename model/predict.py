import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

BEST_THRESHOLD = 0.50

# -------- PATH FIX (VERY IMPORTANT) --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "deepfake_model.h5")

IMG_SIZE = 260



# Load trained model
model = load_model(MODEL_PATH)

from tensorflow.keras.optimizers import Adam

# ================= SAFE ONLINE LEARNING SETUP =================

# Freeze all layers except the last 2 (prevents forgetting)
for layer in model.layers[:-2]:
    layer.trainable = False

# Re-compile with very small learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # VERY SMALL LR
    loss="binary_crossentropy"
)

# =============================================================


def predict_faces(face_folder):
    """
    Predict average fake probability from extracted face images
    """
    predictions = []

    for img_name in os.listdir(face_folder):
        img_path = os.path.join(face_folder, img_name)

        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prob = model.predict(img, verbose=0)[0][0]
        # Ignore uncertain predictions
        if prob < 0.2 or prob > 0.8:
            predictions.append(prob)

    if len(predictions) == 0:
        return None

    return float(np.median(predictions))


def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img, verbose=0)[0][0]
    return float(1 - prob) * 100


# ================= SAFE ONLINE FINE-TUNING =================

def online_fine_tune(face_tensor, label=0):
    """
    Privacy-safe incremental learning
    label: 0 = REAL (only allowed)
    """
    # Ensure correct shape
    X = np.expand_dims(face_tensor, axis=0)
    y = np.array([label])

    # One very small update step
    model.fit(X, y, epochs=1, verbose=0)

# ==========================================================
