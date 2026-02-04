import sys
import os

# 🔑 ADD PROJECT ROOT TO PYTHON PATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import shutil
from flask import Flask, render_template, request
from model.predict import predict_faces, predict_image

from flask import send_from_directory
from report_generator import generate_pdf_report


app = Flask(__name__)

# -------- PATHS --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
FRAME_FOLDER = os.path.join(BASE_DIR, "frames")
FACE_FOLDER = os.path.join(BASE_DIR, "faces")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(FACE_FOLDER, exist_ok=True)

# Load face detector (FREE & OFFLINE)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= FACE TENSOR EXTRACTION (RAM ONLY) =================

def extract_face_tensor_from_image(image_path):
    """
    Extracts a single face as a normalized tensor (260x260) without saving to disk.
    Returns None if no face is detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    # Take the first detected face
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (260, 260))
    face = face / 255.0  # normalize

    return face

# ====================================================================

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Take 1 frame every 10 frames
        if count % 5 == 0:
            frame_path = os.path.join(FRAME_FOLDER, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, frame)

        count += 1

    cap.release()

def extract_faces():
    for img_name in os.listdir(FRAME_FOLDER):
        img_path = os.path.join(FRAME_FOLDER, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (260, 260))
            face_path = os.path.join(FACE_FOLDER, f"{img_name}_{i}.jpg")
            cv2.imwrite(face_path, face)

def extract_face_tensor_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (260, 260))
    face = face / 255.0

    return face

def cleanup_temp_files():
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    shutil.rmtree(FRAME_FOLDER, ignore_errors=True)
    shutil.rmtree(FACE_FOLDER, ignore_errors=True)

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(FRAME_FOLDER, exist_ok=True)
    os.makedirs(FACE_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        # Clean old data (for videos)
        shutil.rmtree(FRAME_FOLDER, ignore_errors=True)
        shutil.rmtree(FACE_FOLDER, ignore_errors=True)
        os.makedirs(FRAME_FOLDER)
        os.makedirs(FACE_FOLDER)

        file = request.files["media"]
        filename = file.filename.lower()
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # -------- IMAGE MODE --------
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            from model.pretrained_predict import predict_pretrained
            from model.predict import BEST_THRESHOLD

            # ---------- CUSTOM MODEL ----------
            truth_custom = predict_image(file_path)   # FIXED
            fake_custom = 1 - (truth_custom / 100)

            # ---------- PRETRAINED MODEL ----------
            fake_pretrained = predict_pretrained(file_path)  # FIXED

            # ---------- SAFE ENSEMBLE ----------
            if fake_pretrained is not None:
                fake_final = (fake_custom + fake_pretrained) / 2
            else:
                fake_final = fake_custom

            truth_probability = round((1 - fake_final) * 100, 2)

            result = {
                "truth": truth_probability,
                "fake": round(100 - truth_probability, 2),
                "custom_fake": round(fake_custom * 100, 2),
                "pretrained_fake": round(fake_pretrained * 100, 2) if fake_pretrained else None
            }

            # -------- SAFE ONLINE LEARNING (REAL ONLY) --------
            consent = request.form.get("consent")
            if consent:
                face_tensor = extract_face_tensor_from_image(file_path)
                if face_tensor is not None:
                    from model.predict import online_fine_tune
                    online_fine_tune(face_tensor, label=0)

            cleanup_temp_files()

        # -------- VIDEO MODE --------
        elif filename.endswith('.mp4'):
            extract_frames(file_path)
            extract_faces()

            from model.pretrained_predict import predict_pretrained_video

            # ---------- CUSTOM MODEL ----------
            fake_custom = predict_faces(FACE_FOLDER)

            # ---------- PRETRAINED MODEL ----------
            fake_pretrained = predict_pretrained_video(FACE_FOLDER)

            # ---------- SAFE ENSEMBLE ----------
            if fake_custom is None and fake_pretrained is None:
                result = "No face detected in video."
            else:
                if fake_pretrained is not None:
                    fake_final = (fake_custom + fake_pretrained) / 2
                else:
                    fake_final = fake_custom

                truth_probability = round((1 - fake_final) * 100, 2)

                result = {
                    "truth": truth_probability,
                    "fake": round(100 - truth_probability, 2),
                    "custom_fake": round(fake_custom * 100, 2),
                    "pretrained_fake": round(fake_pretrained * 100, 2) if fake_pretrained else None
                }

            cleanup_temp_files()

        else:
            result = {
                "truth": 0,
                "fake": 0
            }
            cleanup_temp_files()

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
