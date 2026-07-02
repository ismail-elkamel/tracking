from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kidney_segmentation.models import DEFAULT_MODEL, build_model


def export_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    device_name: str = "cpu",
    opset: int = 18,
) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location=device_name)
    model_name = checkpoint.get("model", DEFAULT_MODEL)
    image_size = int(checkpoint.get("image_size", 512))
    device = torch.device(device_name)

    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a kidney segmentation checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="kidney_segmentation/runs/kidney_model/best.pt")
    parser.add_argument("--output", default="kidney_segmentation/runs/kidney_model/best.onnx")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    output_path = export_checkpoint_to_onnx(args.checkpoint, args.output, args.device, args.opset)
    print(f"Exported ONNX model: {output_path}")


if __name__ == "__main__":
    main()
