from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_pil(image: Image.Image, image_size: int):
    resized = F.resize(image.convert("RGB"), [image_size, image_size], antialias=True)
    tensor = F.to_tensor(resized)
    return F.normalize(tensor, mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()).unsqueeze(0)


def preprocess_numpy(image: Image.Image, image_size: int) -> np.ndarray:
    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def sigmoid(array: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-array))


def overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    image_np = np.asarray(image.convert("RGB")).copy()
    overlay = image_np.copy()
    overlay[mask] = (255, 40, 40)
    output = cv2.addWeighted(image_np, 0.65, overlay, 0.35, 0.0)
    return Image.fromarray(output)


def save_overlay(image: Image.Image, probability: np.ndarray, output_path: str | Path, threshold: float) -> Path:
    original_w, original_h = image.size
    probability = cv2.resize(probability, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    output = overlay_mask(image, probability >= threshold)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)
    return path

