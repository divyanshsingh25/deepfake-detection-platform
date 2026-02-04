import kagglehub
import os
import shutil

# Download dataset
path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")

print("Dataset downloaded at:", path)

REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            src = os.path.join(root, file)

            if "real" in root.lower():
                shutil.copy(src, REAL_DIR)
            elif "fake" in root.lower():
                shutil.copy(src, FAKE_DIR)

print("Images copied into dataset/real and dataset/fake")
