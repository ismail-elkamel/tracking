from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from instrument_segmentation.models import build_model


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = F.resize(image.convert("RGB"), [image_size, image_size], antialias=True)
    tensor = F.to_tensor(resized)
    return F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run instrument segmentation on one image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_name = checkpoint.get("model", "deeplabv3plus_efficientnet_b4")
    image_size = int(checkpoint.get("image_size", 512))
    model = build_model(model_name).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    original_w, original_h = image.size
    tensor = preprocess(image, image_size).to(args.device)
    with torch.inference_mode():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    prob = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    mask = prob >= args.threshold

    image_np = np.asarray(image).copy()
    overlay = image_np.copy()
    overlay[mask] = (255, 40, 40)
    output = cv2.addWeighted(image_np, 0.65, overlay, 0.35, 0.0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(output_path)


if __name__ == "__main__":
    main()

