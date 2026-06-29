from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from instrument_segmentation.inference import overlay_mask, preprocess_numpy, sigmoid


def make_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}.")
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONNX instrument segmentation on a video.")
    parser.add_argument("--onnx", default="instrument_segmentation/runs/instrument_model/best.onnx")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="instrument_segmentation/runs/instrument_video_overlay.mp4")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-mask-video", default=None)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is required for ONNX video inference. Install it with "
            "`python -m pip install onnxruntime` or `python -m pip install onnxruntime-gpu`."
        ) from error

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    session = ort.InferenceSession(args.onnx, providers=ort.get_available_providers())
    input_name = session.get_inputs()[0].name
    overlay_writer = make_writer(Path(args.output), fps, width, height)
    mask_writer = None
    if args.save_mask_video:
        mask_writer = make_writer(Path(args.save_mask_video), fps, width, height)

    try:
        progress = tqdm(total=frame_count if frame_count > 0 else None, desc="infer video")
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            input_array = preprocess_numpy(image, args.image_size)
            logits = session.run(None, {input_name: input_array})[0]
            probability = sigmoid(np.asarray(logits)[0, 0])
            probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = probability >= args.threshold

            overlay_rgb = np.asarray(overlay_mask(image, mask))
            overlay_writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

            if mask_writer is not None:
                mask_frame = (mask.astype(np.uint8) * 255)
                mask_writer.write(cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR))

            progress.update(1)
        progress.close()
    finally:
        cap.release()
        overlay_writer.release()
        if mask_writer is not None:
            mask_writer.release()

    print(f"Overlay video: {args.output}")
    if args.save_mask_video:
        print(f"Mask video: {args.save_mask_video}")


if __name__ == "__main__":
    main()

