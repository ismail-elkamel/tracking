from __future__ import annotations

import argparse

import torch
from PIL import Image

from instrument_segmentation.inference import preprocess_pil, save_overlay
from instrument_segmentation.models import build_model


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
    tensor = preprocess_pil(image, image_size).to(args.device)
    with torch.inference_mode():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    save_overlay(image, prob, args.output, args.threshold)


if __name__ == "__main__":
    main()
