"""
utils/preprocessing.py
=======================
Handles all data preprocessing for DeepShield:
- Frame extraction from videos
- Image resizing and normalization
- Data augmentation pipeline
- Dataset splitting utilities
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE = 224          # ResNet/EfficientNet input size
FRAMES_PER_VIDEO = 20   # How many frames to sample per video
MEAN = [0.485, 0.456, 0.406]   # ImageNet mean
STD  = [0.229, 0.224, 0.225]   # ImageNet std


# ─────────────────────────────────────────────
# 1. Frame Extraction from Video
# ─────────────────────────────────────────────
def extract_frames(video_path: str, output_dir: str, n_frames: int = FRAMES_PER_VIDEO) -> list:
    """
    Extract evenly-spaced frames from a video file.

    Args:
        video_path  : Path to input video (.mp4, .avi, etc.)
        output_dir  : Directory to save extracted frames
        n_frames    : Number of frames to extract

    Returns:
        List of saved frame file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Video has 0 frames: {video_path}")

    # Evenly sample frame indices
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    saved_paths = []
    video_name = Path(video_path).stem

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = os.path.join(output_dir, f"{video_name}_frame_{idx:05d}.jpg")
        cv2.imwrite(out_path, frame)
        saved_paths.append(out_path)

    cap.release()
    return saved_paths


def extract_frames_from_folder(video_folder: str, frames_folder: str, label: str):
    """
    Extract frames from all videos in a folder.
    label: 'real' or 'fake'
    """
    out_dir = os.path.join(frames_folder, label)
    os.makedirs(out_dir, exist_ok=True)

    video_files = list(Path(video_folder).glob("*.mp4")) + \
                  list(Path(video_folder).glob("*.avi")) + \
                  list(Path(video_folder).glob("*.mov"))

    print(f"Extracting frames for {label} ({len(video_files)} videos)...")
    for vf in tqdm(video_files):
        try:
            extract_frames(str(vf), out_dir, FRAMES_PER_VIDEO)
        except Exception as e:
            print(f"  Skipping {vf.name}: {e}")


# ─────────────────────────────────────────────
# 2. Augmentation Pipelines
# ─────────────────────────────────────────────
def get_train_transforms():
    """Augmentation-heavy pipeline for training."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Simple resize + normalize for validation/test."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_inference_transforms():
    """PyTorch transforms for single-image inference."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ─────────────────────────────────────────────
# 3. Custom Dataset Class
# ─────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset that loads face-cropped images.

    Expected folder structure:
        root/
          real/  *.jpg
          fake/  *.jpg

    Labels: real=0, fake=1
    """

    def __init__(self, root_dir: str, transform=None):
        self.samples = []   # list of (image_path, label)
        self.transform = transform

        class_map = {"real": 0, "fake": 1}
        for cls, lbl in class_map.items():
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f"[Warning] Missing folder: {cls_dir}")
                continue
            for img_file in Path(cls_dir).glob("*.jpg"):
                self.samples.append((str(img_file), lbl))
            for img_file in Path(cls_dir).glob("*.png"):
                self.samples.append((str(img_file), lbl))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        print(f"Dataset loaded: {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            # Return a black image on failure (rare edge case)
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


# ─────────────────────────────────────────────
# 4. DataLoader Factory
# ─────────────────────────────────────────────
def get_dataloaders(train_dir: str, val_dir: str, test_dir: str,
                    batch_size: int = 32, num_workers: int = 4):
    """
    Returns train, val, test DataLoaders.
    """
    train_ds = DeepfakeDataset(train_dir, transform=get_train_transforms())
    val_ds   = DeepfakeDataset(val_dir,   transform=get_val_transforms())
    test_ds  = DeepfakeDataset(test_dir,  transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 5. Dataset Split Utility
# ─────────────────────────────────────────────
def split_dataset(source_dir: str, output_dir: str,
                  train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                  seed: int = 42):
    """
    Split images from source_dir (real/ fake/) into
    output_dir/train/, output_dir/val/, output_dir/test/.

    Ratios must sum to 1.0.
    """
    import shutil, random
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(seed)
    np.random.seed(seed)

    for cls in ["real", "fake"]:
        cls_dir = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        files = list(Path(cls_dir).glob("*.jpg")) + list(Path(cls_dir).glob("*.png"))
        random.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        splits = {
            "train": files[:n_train],
            "val":   files[n_train:n_train + n_val],
            "test":  files[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            dest = os.path.join(output_dir, split_name, cls)
            os.makedirs(dest, exist_ok=True)
            for f in tqdm(split_files, desc=f"{split_name}/{cls}"):
                shutil.copy(str(f), os.path.join(dest, f.name))

        print(f"{cls}: {n_train} train | {n_val} val | {n - n_train - n_val} test")


# ─────────────────────────────────────────────
# CLI Usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Example: Extract frames from dataset videos
    # extract_frames_from_folder("dataset/videos/real", "dataset/frames", "real")
    # extract_frames_from_folder("dataset/videos/fake", "dataset/frames", "fake")

    # Then split:
    # split_dataset("dataset/frames", "dataset/split")
    print("Preprocessing utilities ready. Import and use in your pipeline.")
