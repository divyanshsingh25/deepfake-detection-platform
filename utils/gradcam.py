"""
utils/gradcam.py
================
Gradient-weighted Class Activation Mapping (Grad-CAM).
Produces a heatmap showing WHICH regions of the face
the model focused on when predicting Real vs Fake.

Works with ResNet50 and EfficientNet-B0.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for CNNs.

    Usage:
        cam = GradCAM(model, target_layer=model.layer4[-1])  # ResNet50
        heatmap, overlay = cam(input_tensor, class_idx=1)    # 1 = fake
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks on the target layer
        self._register_hooks()

    def _register_hooks(self):
        """Attach forward and backward hooks to the target layer."""

        def forward_hook(module, input, output):
            # Store feature maps from forward pass
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Store gradients flowing back through this layer
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor,
                 class_idx: int = None) -> tuple:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor : [1, 3, H, W] tensor (normalized)
            class_idx    : Target class (0=real, 1=fake). If None, uses predicted class.

        Returns:
            heatmap : [H, W] numpy array (0–1 values, upsampled to input size)
            overlay : [H, W, 3] numpy uint8 image (heatmap blended on face crop)
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        # ── Forward pass ──────────────────────────────
        logits = self.model(input_tensor)           # [1, 2]
        probs  = F.softmax(logits, dim=1)           # [1, 2]

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # ── Backward pass ─────────────────────────────
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # ── Compute Grad-CAM ──────────────────────────
        # gradients  : [1, C, H, W]
        # activations: [1, C, H, W]
        gradients   = self.gradients[0]      # [C, H, W]
        activations = self.activations[0]    # [C, H, W]

        # Global average pooling over spatial dims → weight per channel
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Weighted sum of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU (keep only positive contributions)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input image size
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(cam, (input_w, input_h))

        # Create colour overlay
        overlay = self._make_overlay(input_tensor, heatmap)

        return heatmap, overlay, probs[0, class_idx].item()

    @staticmethod
    def _make_overlay(input_tensor: torch.Tensor,
                      heatmap: np.ndarray) -> np.ndarray:
        """
        Blend Grad-CAM heatmap onto the original face image.

        Returns RGB uint8 image.
        """
        # Denormalize input tensor to 0-255 for display
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        img = input_tensor.squeeze(0).permute(1, 2, 0).numpy()
        img = std * img + mean                          # denormalize
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # Convert heatmap to jet colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Blend: 60% original, 40% heatmap
        overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        return overlay


# ─────────────────────────────────────────────
# Target Layer Helpers
# ─────────────────────────────────────────────
def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """
    Return the appropriate last convolutional layer for Grad-CAM.
    """
    if model_name == "resnet50":
        return model.layer4[-1]           # last residual block
    elif model_name in ("efficientnet_b0", "efficientnet"):
        # torchvision EfficientNet
        return model.features[-1]         # last feature block
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         "Manually pass target_layer to GradCAM.")


# ─────────────────────────────────────────────
# Convenience: Run Grad-CAM on a PIL Image
# ─────────────────────────────────────────────
def run_gradcam_on_image(model, model_name: str,
                         pil_image: Image.Image,
                         transform,
                         device: str = "cpu",
                         class_idx: int = None):
    """
    Full pipeline: PIL Image → tensor → Grad-CAM → overlay image.

    Returns:
        heatmap  : numpy [H, W]
        overlay  : numpy [H, W, 3] uint8 RGB
        prob     : float, confidence for the target class
        pred_cls : int, 0=real / 1=fake
    """
    tensor = transform(pil_image).unsqueeze(0).to(device)
    target_layer = get_target_layer(model, model_name)

    cam = GradCAM(model, target_layer)
    with torch.enable_grad():
        heatmap, overlay, prob = cam(tensor, class_idx)

    # Also get final prediction
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        pred_cls = logits.argmax(dim=1).item()

    return heatmap, overlay, prob, pred_cls
