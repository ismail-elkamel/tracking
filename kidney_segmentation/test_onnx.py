from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from kidney_segmentation.inference import preprocess_numpy, save_overlay, sigmoid


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in suffixes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test an ONNX kidney segmentation model on new images.")
    parser.add_argument("--onnx", default="kidney_segmentation/runs/kidney_model/best.onnx")
    parser.add_argument("--input", required=True, help="Image file or folder of images.")
    parser.add_argument("--output-dir", default="kidney_segmentation/runs/onnx_predictions")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is required for ONNX testing. Install it with "
            "`python -m pip install onnxruntime` or `python -m pip install onnxruntime-gpu`."
        ) from error

    image_paths = iter_images(Path(args.input))
    if not image_paths:
        raise RuntimeError(f"No images found in {args.input!r}.")

    session = ort.InferenceSession(args.onnx, providers=ort.get_available_providers())
    input_name = session.get_inputs()[0].name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        input_array = preprocess_numpy(image, args.image_size)
        logits = session.run(None, {input_name: input_array})[0]
        probability = sigmoid(np.asarray(logits)[0, 0])
        output_path = output_dir / f"{image_path.stem}_kidney_overlay.png"
        save_overlay(image, probability, output_path, args.threshold)
        print(output_path)


if __name__ == "__main__":
    main()

