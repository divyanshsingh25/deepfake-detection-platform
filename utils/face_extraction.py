"""
utils/face_extraction.py
========================
Face detection and cropping using MTCNN (facenet-pytorch).
Works on both single images and video frames.
Saves cropped face regions for model input.
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("[Warning] facenet-pytorch not installed. Using OpenCV Haar cascade fallback.")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FACE_SIZE = 224
MARGIN = 20   # pixels to pad around detected face box


# ─────────────────────────────────────────────
# 1. Initialise Detector
# ─────────────────────────────────────────────
def get_face_detector(device="cpu"):
    """
    Returns an MTCNN detector.
    Falls back to None if library unavailable (use OpenCV Haar then).
    """
    if MTCNN_AVAILABLE:
        detector = MTCNN(
            image_size=FACE_SIZE,
            margin=MARGIN,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],  # P/R/O net thresholds
            keep_all=False,               # return only highest-confidence face
            device=device,
            post_process=False,           # return raw pixel values (0-255)
        )
        return detector
    return None


# OpenCV fallback
_haar_cascade = None

def _get_haar():
    global _haar_cascade
    if _haar_cascade is None:
        xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _haar_cascade = cv2.CascadeClassifier(xml_path)
    return _haar_cascade


# ─────────────────────────────────────────────
# 2. Extract Face from Single Image (numpy array)
# ─────────────────────────────────────────────
def extract_face(image_rgb: np.ndarray, detector=None) -> np.ndarray:
    """
    Detect and crop the primary face from an RGB image.

    Args:
        image_rgb : H×W×3 numpy array in RGB order
        detector  : MTCNN detector or None (uses Haar cascade)

    Returns:
        224×224×3 numpy array (face crop) or None if no face found
    """
    h, w = image_rgb.shape[:2]

    if detector is not None:
        # MTCNN path
        pil_img = Image.fromarray(image_rgb)
        face_tensor = detector(pil_img)   # returns tensor or None
        if face_tensor is not None:
            # face_tensor: [C, H, W] with raw pixel values (0-255) float
            face_np = face_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
            face_resized = cv2.resize(face_np, (FACE_SIZE, FACE_SIZE))
            return face_resized
    else:
        # OpenCV Haar cascade fallback
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        cascade = _get_haar()
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(40, 40))
        if len(faces) > 0:
            # Pick the largest face
            areas = [fw * fh for (_, _, fw, fh) in faces]
            x, y, fw, fh = faces[np.argmax(areas)]
            # Add margin
            x1 = max(0, x - MARGIN)
            y1 = max(0, y - MARGIN)
            x2 = min(w, x + fw + MARGIN)
            y2 = min(h, y + fh + MARGIN)
            face_crop = image_rgb[y1:y2, x1:x2]
            face_resized = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))
            return face_resized

    return None   # No face detected


# ─────────────────────────────────────────────
# 3. Process a Folder of Images
# ─────────────────────────────────────────────
def process_image_folder(input_dir: str, output_dir: str, detector=None):
    """
    Crop faces from all images in input_dir and save to output_dir.
    Preserves real/fake subfolder structure.
    """
    for cls in ["real", "fake"]:
        src = os.path.join(input_dir, cls)
        dst = os.path.join(output_dir, cls)
        if not os.path.isdir(src):
            continue
        os.makedirs(dst, exist_ok=True)

        files = list(Path(src).glob("*.jpg")) + list(Path(src).glob("*.png"))
        skipped = 0

        for f in tqdm(files, desc=f"Face crop [{cls}]"):
            img_bgr = cv2.imread(str(f))
            if img_bgr is None:
                skipped += 1
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face = extract_face(img_rgb, detector)

            if face is None:
                # No face found: save resized original as fallback
                face = cv2.resize(img_rgb, (FACE_SIZE, FACE_SIZE))
                skipped += 1

            save_path = os.path.join(dst, f.name)
            cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        print(f"  {cls}: {len(files)} images | {skipped} no-face fallbacks")


# ─────────────────────────────────────────────
# 4. Extract Face from a Single File Path
# ─────────────────────────────────────────────
def extract_face_from_path(image_path: str, detector=None) -> np.ndarray:
    """
    Convenience wrapper: load image → extract face → return RGB numpy array.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face = extract_face(img_rgb, detector)
    if face is None:
        # Return full image resized if no face detected
        face = cv2.resize(img_rgb, (FACE_SIZE, FACE_SIZE))
    return face


# ─────────────────────────────────────────────
# 5. Real-time Webcam Face Extraction
# ─────────────────────────────────────────────
def get_face_from_frame(frame_bgr: np.ndarray, detector=None) -> np.ndarray:
    """
    Used in real-time webcam pipeline.
    frame_bgr: OpenCV BGR frame.
    Returns RGB face crop or None.
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return extract_face(img_rgb, detector)


if __name__ == "__main__":
    detector = get_face_detector(device="cpu")
    print("MTCNN detector ready:", detector is not None)
    print("Face extraction utilities loaded.")
