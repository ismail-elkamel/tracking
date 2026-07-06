from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track app prompts with TAPIR/BootsTAPIR.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model-profile", default="bootstapir", choices=["tapir", "bootstapir"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-path", default="external/tapnet")
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--hide-lost-points", action="store_true")
    return parser.parse_args()


def add_tapnet_to_path(repo_path: str) -> None:
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(
            f"TAPNet repo not found: {path}\n"
            "Install it with: git clone https://github.com/google-deepmind/tapnet external/tapnet"
        )
    sys.path.insert(0, str(path))


def import_tapir_model(repo_path: str):
    add_tapnet_to_path(repo_path)
    try:
        from tapnet.torch import tapir_model
    except ImportError as error:
        raise RuntimeError(
            "Could not import TAPNet PyTorch TAPIR. Install dependencies with:\n"
            "python -m pip install -e external/tapnet[torch]\n\n"
            f"Original error: {error}"
        ) from error
    return tapir_model


def read_prompts(path: str | Path) -> tuple[list[np.ndarray], list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tracks: list[np.ndarray] = []
    labels: list[str] = []
    for index, annotation in enumerate(payload.get("annotations", [])):
        points = np.asarray(annotation.get("points_xy", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            continue
        tracks.append(points)
        labels.append(annotation.get("label") or f"annotation {index + 1}")
    if not tracks:
        raise RuntimeError("Prompt file does not contain any valid points.")
    return tracks, labels


def flatten_tracks(tracks: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    sizes = [len(track) for track in tracks]
    return np.concatenate(tracks, axis=0).astype(np.float32), sizes


def regroup_points(points: np.ndarray, visible: np.ndarray, sizes: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped: list[tuple[np.ndarray, np.ndarray]] = []
    offset = 0
    for size in sizes:
        grouped.append((points[offset : offset + size], visible[offset : offset + size]))
        offset += size
    return grouped


def read_video_interval(path: str, start_frame: int, end_frame: int) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
        for _ in range(max(0, end_frame - start_frame + 1)):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Could not read frames {start_frame}-{end_frame} from {path}")
    return frames, fps


def resize_frames(frames: list[np.ndarray], resize_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = frames[0].shape[:2]
    size = max(64, int(resize_size))
    resized = [
        cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
    return np.stack(resized), (width, height)


def points_to_tapir_queries(points_xy: np.ndarray, source_size: tuple[int, int], resize_size: int) -> np.ndarray:
    width, height = source_size
    queries = np.zeros((len(points_xy), 3), dtype=np.float32)
    queries[:, 0] = 0.0
    queries[:, 1] = points_xy[:, 1] * float(resize_size) / max(float(height), 1.0)
    queries[:, 2] = points_xy[:, 0] * float(resize_size) / max(float(width), 1.0)
    return queries


def postprocess_visibility(occlusions: torch.Tensor, expected_dist: torch.Tensor, threshold: float) -> torch.Tensor:
    confidence = (1.0 - torch.sigmoid(occlusions)) * (1.0 - torch.sigmoid(expected_dist))
    return confidence > float(threshold)


def load_model(args: argparse.Namespace, device: torch.device):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"{args.model_profile} checkpoint not found: {checkpoint_path}\n"
            "Use the app sidebar download button or download it manually."
        )
    tapir_model = import_tapir_model(args.repo_path)
    model = tapir_model.TAPIR(pyramid_level=1)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def run_tapir(
    frames_rgb: list[np.ndarray],
    points_xy: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    resized_frames, source_size = resize_frames(frames_rgb, args.resize_size)
    queries = points_to_tapir_queries(points_xy, source_size, args.resize_size)
    model = load_model(args, device)

    video_tensor = torch.from_numpy(resized_frames).to(device=device, dtype=torch.float32)
    video_tensor = video_tensor / 255.0 * 2.0 - 1.0
    query_tensor = torch.from_numpy(queries).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        outputs = model(
            video_tensor[None],
            query_tensor[None],
            query_chunk_size=max(1, int(args.query_chunk_size)),
        )
    tracks = outputs["tracks"][0].detach().cpu().numpy().astype(np.float32)
    visible = postprocess_visibility(
        outputs["occlusion"][0],
        outputs["expected_dist"][0],
        args.visibility_threshold,
    ).detach().cpu().numpy().astype(bool)

    width, height = source_size
    tracks[..., 0] *= float(width) / float(args.resize_size)
    tracks[..., 1] *= float(height) / float(args.resize_size)
    return tracks, visible


def draw_tracks(
    frame_rgb: np.ndarray,
    points: np.ndarray,
    visible: np.ndarray,
    sizes: list[int],
    labels: list[str],
    hide_lost: bool,
) -> np.ndarray:
    output = frame_rgb.copy()
    colors = [
        (255, 72, 92),
        (28, 167, 236),
        (20, 184, 124),
        (255, 177, 66),
        (159, 122, 234),
        (245, 101, 101),
    ]
    for index, (group, group_visible) in enumerate(regroup_points(points, visible, sizes)):
        if hide_lost:
            group = group[group_visible]
        if len(group) == 0:
            continue
        color = colors[index % len(colors)]
        pts = np.round(group).astype(np.int32)
        if len(pts) == 1:
            cv2.circle(output, tuple(pts[0]), 7, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(output, tuple(pts[0]), 11, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        else:
            closed = len(pts) >= 3
            cv2.polylines(output, [pts], closed, color, 3, lineType=cv2.LINE_AA)
            for point in pts:
                cv2.circle(output, tuple(point), 5, color, -1, lineType=cv2.LINE_AA)
        label = labels[index] if index < len(labels) else f"annotation {index + 1}"
        if label and not label.startswith("obj "):
            cv2.putText(
                output,
                label,
                (int(pts[0][0]) + 8, int(pts[0][1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return output


def write_video(
    output_path: str | Path,
    frames_rgb: list[np.ndarray],
    tracks: np.ndarray,
    visible: np.ndarray,
    sizes: list[int],
    labels: list[str],
    fps: float,
    hide_lost: bool,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    try:
        for frame_rgb, frame_points, frame_visible in zip(frames_rgb, tracks, visible):
            drawn = draw_tracks(frame_rgb, frame_points, frame_visible, sizes, labels, hide_lost)
            writer.write(cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"TAPIR finished but did not create output video: {path}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    tracks, labels = read_prompts(args.prompts)
    points_xy, sizes = flatten_tracks(tracks)
    frames_rgb, fps = read_video_interval(args.video, args.start_frame, args.end_frame)
    predicted, visible = run_tapir(frames_rgb, points_xy, args, device)
    write_video(args.output, frames_rgb, predicted, visible, sizes, labels, fps, args.hide_lost_points)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(str(error)) from error
