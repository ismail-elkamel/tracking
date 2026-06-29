from __future__ import annotations

import json
import shutil
import shlex
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


MODEL_DIR = Path("models")
LITETRACKER_DIR = Path("external/lite-tracker")
LITETRACKER_WEIGHTS_URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth"

OPENCV_TRACKER = "OpenCV Lucas-Kanade"
COTRACKER_TRACKER = "CoTracker3 Online"
COTRACKER_OFFLINE_TRACKER = "CoTracker3 Offline"
LITETRACKER_TRACKER = "LiteTracker"
SAM2_TRACKER = "SAM2"
SURGISAM2_TRACKER = "SurgiSAM2"
SAM3_TRACKER = "SAM3"
MEDSAM2_TRACKER = "MedSAM2"
CUDA_DEVICE = "CUDA GPU"
CPU_DEVICE = "CPU"


def cuda_is_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def cuda_summary() -> str:
    import torch

    if not torch.cuda.is_available():
        return "CUDA GPU is not visible to PyTorch right now."
    return f"CUDA GPU: {torch.cuda.get_device_name(0)}"


def resolve_torch_device(device_name: str):
    import torch

    if device_name == CUDA_DEVICE:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU was selected, but PyTorch cannot see a CUDA device. "
                "Check the NVIDIA driver, CUDA runtime, and whether this environment has GPU access. "
                "For now, choose CPU to test the tracker."
            )
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cpu")


def default_litetracker_weights_path() -> Path:
    local_path = MODEL_DIR / "scaled_online.pth"
    if local_path.exists():
        return local_path
    cached_path = Path.home() / ".cache/torch/hub/checkpoints/scaled_online.pth"
    if cached_path.exists():
        return cached_path
    return local_path


def download_litetracker_weights(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.pth")
    urllib.request.urlretrieve(LITETRACKER_WEIGHTS_URL, tmp_path)
    tmp_path.replace(path)


def tracker_slug(tracker_name: str) -> str:
    return tracker_name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def default_external_command(tracker_name: str) -> str:
    if tracker_name == SAM2_TRACKER:
        return (
            "python external/Surgical-SAM-2/infer_prompts.py "
            "--video {video} --start-frame {start_frame} --prompts {prompts} "
            "--output {output} --device {device} --model-profile sam2 {freeze_lost_points}"
        )
    if tracker_name == SURGISAM2_TRACKER:
        return (
            "python external/Surgical-SAM-2/infer_prompts.py "
            "--video {video} --start-frame {start_frame} --prompts {prompts} "
            "--output {output} --device {device} --model-profile surgisam2 {freeze_lost_points}"
        )
    if tracker_name == SAM3_TRACKER:
        return (
            "conda run -n track_env python external/SAM3/infer_prompts.py "
            "--video {video} --start-frame {start_frame} --prompts {prompts} "
            "--output {output} --device {device} {freeze_lost_points}"
        )
    if tracker_name == MEDSAM2_TRACKER:
        return (
            "python external/MedSAM2/infer_prompts.py "
            "--video {video} --start-frame {start_frame} --prompts {prompts} "
            "--output {output} --device {device} {freeze_lost_points}"
        )
    return ""


def external_tracker_setup_instructions(tracker_name: str) -> str:
    if tracker_name == SAM2_TRACKER:
        return (
            "SAM2 uses the generic SAM2.1 checkpoint "
            "`external/Surgical-SAM-2/checkpoints/sam2.1_hiera_small.pt`."
        )
    if tracker_name == SURGISAM2_TRACKER:
        return (
            "SurgiSAM2 requires the fine-tuned checkpoint "
            "`external/Surgical-SAM-2/checkpoints/Curated400_checkpoint_26.pt`. "
            "It will not silently fall back to generic SAM2."
        )
    if tracker_name == SAM3_TRACKER:
        return (
            "SAM3 uses Meta's `facebookresearch/sam3` repo. It requires a separate "
            "conda environment, CUDA, and accepted/authenticated Hugging Face "
            "access to the SAM3 or SAM3.1 checkpoints."
        )
    if tracker_name == MEDSAM2_TRACKER:
        return (
            "MedSAM2 uses the bowang-lab/MedSAM2 video predictor. The app turns "
            "your first-frame point/region prompts into a mask prompt, then "
            "propagates that mask through the video. Download MedSAM2 checkpoints "
            "with `bash external/MedSAM2/download.sh` before running."
        )
    return "Install the external model code, then update the sidebar command."


def write_prompt_file(
    prompt_path: str | Path,
    tracks: list[np.ndarray],
    labels: list[str],
    start_frame: int,
) -> Path:
    path = Path(prompt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "start_frame": int(start_frame),
        "annotations": [
            {
                "label": labels[index] if index < len(labels) else f"annotation {index + 1}",
                "points_xy": track.astype(float).round(3).tolist(),
                "kind": "point" if len(track) == 1 else "region",
            }
            for index, track in enumerate(tracks)
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def draw_tracks(frame_rgb: np.ndarray, tracks: list[np.ndarray], labels: list[str]) -> np.ndarray:
    output = frame_rgb.copy()
    colors = [
        (255, 72, 92),
        (28, 167, 236),
        (20, 184, 124),
        (255, 177, 66),
        (159, 122, 234),
        (245, 101, 101),
    ]

    for index, points in enumerate(tracks):
        color = colors[index % len(colors)]
        pts = np.round(points).astype(int)
        if len(pts) == 1:
            cv2.circle(output, tuple(pts[0]), 7, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(output, tuple(pts[0]), 11, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        else:
            closed = len(pts) >= 3
            cv2.polylines(output, [pts], closed, color, 3, lineType=cv2.LINE_AA)
            for point in pts:
                cv2.circle(output, tuple(point), 5, color, -1, lineType=cv2.LINE_AA)
        label = labels[index] if index < len(labels) else f"annotation {index + 1}"
        if len(pts) and label and not label.startswith("grid "):
            x, y = pts[0]
            cv2.putText(
                output,
                label,
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return output


def flatten_tracks(tracks: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    group_sizes = [len(track) for track in tracks]
    if not tracks:
        return np.empty((0, 2), dtype=np.float32), group_sizes
    return np.concatenate(tracks, axis=0).astype(np.float32), group_sizes


def regroup_points(points: np.ndarray, group_sizes: list[int]) -> list[np.ndarray]:
    grouped: list[np.ndarray] = []
    offset = 0
    for size in group_sizes:
        grouped.append(points[offset : offset + size].astype(np.float32))
        offset += size
    return grouped


def point_in_frame(point: np.ndarray, frame_rgb: np.ndarray) -> bool:
    height, width = frame_rgb.shape[:2]
    x, y = point
    return 0.0 <= float(x) < float(width) and 0.0 <= float(y) < float(height)


def filter_visible_points(
    points: np.ndarray,
    last_valid_points: np.ndarray,
    visible_points: np.ndarray,
    frame_rgb: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = points.astype(np.float32, copy=True)
    last_valid_points = last_valid_points.astype(np.float32, copy=True)
    visible_points = visible_points.astype(bool, copy=True)
    if valid_mask is None:
        valid_mask = np.ones(len(points), dtype=bool)
    else:
        valid_mask = valid_mask.astype(bool, copy=False)

    height, width = frame_rgb.shape[:2]
    in_frame = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < float(width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < float(height))
    )
    visible_points = valid_mask & in_frame
    last_valid_points[visible_points] = points[visible_points]
    return last_valid_points.copy(), last_valid_points, visible_points


def regroup_visible_points(
    points: np.ndarray,
    group_sizes: list[int],
    visible_points: np.ndarray,
) -> list[np.ndarray]:
    grouped: list[np.ndarray] = []
    offset = 0
    for size in group_sizes:
        group = points[offset : offset + size]
        group_visible = visible_points[offset : offset + size]
        grouped.append(group[group_visible].astype(np.float32))
        offset += size
    return grouped


def resize_for_model(frame_rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = frame_rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return frame_rgb, scale
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame_rgb, new_size, interpolation=cv2.INTER_AREA), scale


def sync_to_video_clock(start_time: float, relative_frame: int, fps: float) -> None:
    target_elapsed = relative_frame / max(fps, 1.0)
    actual_elapsed = time.perf_counter() - start_time
    if target_elapsed > actual_elapsed:
        time.sleep(target_elapsed - actual_elapsed)


def open_output_writer(output_path: str | Path | None, frame_rgb: np.ndarray, fps: float):
    if output_path is None:
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_rgb.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer


def write_output_frame(writer, frame_rgb: np.ndarray) -> None:
    if writer is not None:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def emit_tracked_frame(
    frame_rgb: np.ndarray,
    tracks: list[np.ndarray],
    labels: list[str],
    output_writer,
    show_live_preview: bool,
    frame_placeholder,
) -> None:
    drawn = draw_tracks(frame_rgb, tracks, labels)
    write_output_frame(output_writer, drawn)
    if show_live_preview:
        frame_placeholder.image(drawn, channels="RGB", use_container_width=True)


def _resize_tile(frame_bgr: np.ndarray, tile_size: tuple[int, int]) -> np.ndarray:
    tile_width, tile_height = tile_size
    return cv2.resize(frame_bgr, (tile_width, tile_height), interpolation=cv2.INTER_AREA)


def _label_tile(tile_bgr: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    output = tile_bgr.copy()
    banner_height = max(34, int(output.shape[0] * 0.075))
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], banner_height), (12, 14, 18), -1)
    output = cv2.addWeighted(overlay, 0.72, output, 0.28, 0)
    cv2.rectangle(output, (0, 0), (output.shape[1], banner_height), color, 3)

    font_scale = max(0.48, min(0.78, output.shape[1] / 700))
    thickness = 2
    text = label
    while text:
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if text_width <= output.shape[1] - 24:
            break
        text = text[:-1]
    if text != label and len(text) > 3:
        text = text[:-3] + "..."
    cv2.putText(
        output,
        text,
        (12, int(banner_height * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return output


def create_comparison_collage(
    videos: list[tuple[str, str | Path]],
    output_path: str | Path,
    fps: float,
    tile_width: int,
    show_live_preview: bool,
    frame_placeholder,
    status_placeholder,
) -> Path:
    if len(videos) < 2:
        raise RuntimeError("Choose at least two completed model outputs to build a comparison collage.")

    captures = []
    try:
        for label, path in videos:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open comparison video for {label}: {path}")
            captures.append((label, Path(path), cap))

        first_ok, first_frame = captures[0][2].read()
        if not first_ok or first_frame is None:
            raise RuntimeError(f"Could not read first frame from {captures[0][1]}")
        captures[0][2].set(cv2.CAP_PROP_POS_FRAMES, 0)

        source_height, source_width = first_frame.shape[:2]
        tile_width = max(160, int(tile_width))
        tile_height = max(120, int(round(tile_width * source_height / max(source_width, 1))))
        if tile_width % 2:
            tile_width += 1
        if tile_height % 2:
            tile_height += 1
        tile_size = (tile_width, tile_height)
        cols = int(np.ceil(np.sqrt(len(captures))))
        rows = int(np.ceil(len(captures) / cols))
        collage_size = (cols * tile_width, rows * tile_height)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(float(fps), 1.0),
            collage_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create comparison collage: {output_path}")

        colors = [
            (92, 72, 255),
            (236, 167, 28),
            (124, 184, 20),
            (66, 177, 255),
            (234, 122, 159),
            (101, 101, 245),
        ]
        last_frames = [None for _ in captures]
        active = [True for _ in captures]
        frame_index = 0
        start_time = time.perf_counter()

        try:
            while True:
                any_read = False
                for index, (_, _, cap) in enumerate(captures):
                    if not active[index]:
                        continue
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        last_frames[index] = frame
                        any_read = True
                    else:
                        active[index] = False

                if not any_read and frame_index > 0:
                    break
                if not any_read:
                    raise RuntimeError("None of the comparison videos produced a frame.")

                collage = np.zeros((collage_size[1], collage_size[0], 3), dtype=np.uint8)
                for index, (label, _, _) in enumerate(captures):
                    row = index // cols
                    col = index % cols
                    frame = last_frames[index]
                    if frame is None:
                        tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
                    else:
                        tile = _resize_tile(frame, tile_size)
                    tile = _label_tile(tile, label, colors[index % len(colors)])
                    y1 = row * tile_height
                    x1 = col * tile_width
                    collage[y1 : y1 + tile_height, x1 : x1 + tile_width] = tile

                writer.write(collage)
                if show_live_preview:
                    frame_placeholder.image(
                        cv2.cvtColor(collage, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )
                    sync_to_video_clock(start_time, frame_index, fps)
                status_placeholder.caption(f"Building comparison collage frame {frame_index + 1}")
                frame_index += 1
        finally:
            writer.release()

        if frame_index == 0 or not output_path.exists():
            raise RuntimeError("Comparison collage finished without creating an output video.")
        return output_path
    finally:
        for _, _, cap in captures:
            cap.release()


@st.cache_resource(show_spinner=False)
def load_cotracker3_online(device_name: str):
    import torch

    device = resolve_torch_device(device_name)
    try:
        model = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_online",
            trust_repo=True,
        )
    except TypeError:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(device)
    model.eval()
    return model, device


@st.cache_resource(show_spinner=False)
def load_cotracker3_offline(device_name: str):
    import torch

    device = resolve_torch_device(device_name)
    try:
        model = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline",
            trust_repo=True,
        )
    except TypeError:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    model.eval()
    return model, device


@st.cache_resource(show_spinner=False)
def load_litetracker(weights_path: str, device_name: str):
    import torch

    repo_path = LITETRACKER_DIR.resolve()
    checkpoint_path = Path(weights_path).expanduser().resolve()
    if not repo_path.exists():
        raise RuntimeError(
            f"LiteTracker repo not found at {repo_path}. "
            "Clone https://github.com/ImFusionGmbH/lite-tracker there first."
        )
    if not checkpoint_path.exists():
        raise RuntimeError(f"LiteTracker weights not found: {checkpoint_path}")

    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from src.lite_tracker import LiteTracker

    device = resolve_torch_device(device_name)
    model = LiteTracker()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device


def track_with_cotracker3_online(
    path: str,
    start_frame: int,
    end_frame: int,
    tracks: list[np.ndarray],
    labels: list[str],
    fps: float,
    model_max_side: int,
    device_name: str,
    frame_placeholder,
    status_placeholder,
    output_path: str | Path | None = None,
    show_live_preview: bool = False,
    freeze_lost: bool = False,
) -> Path | None:
    import torch

    flat_points, group_sizes = flatten_tracks(tracks)
    if len(flat_points) == 0:
        st.error("No points to track.")
        return

    with st.spinner("Loading CoTracker3 Online..."):
        model, device = load_cotracker3_online(device_name)

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    window: list[tuple[int, np.ndarray, np.ndarray]] = []
    queries = None
    is_first_step = True
    last_displayed_frame = start_frame
    last_processed_frame = start_frame - 1
    start_time = time.perf_counter()
    step = int(getattr(model, "step", 8) or 8)
    model_scale = 1.0
    output_writer = None
    saved_path = Path(output_path) if output_path else None
    last_valid_points = flat_points.copy()
    visible_points = np.ones(len(flat_points), dtype=bool)

    def process_window() -> None:
        nonlocal is_first_step, last_displayed_frame, last_processed_frame, last_valid_points, visible_points
        if not window:
            return

        chunk = window[-step * 2 :]
        model_frames = [item[1] for item in chunk]
        video_chunk = (
            torch.from_numpy(np.stack(model_frames))
            .to(device=device, dtype=torch.float32)
            .permute(0, 3, 1, 2)[None]
        )

        kwargs = {
            "video_chunk": video_chunk,
            "is_first_step": is_first_step,
            "grid_size": 0,
        }
        if is_first_step:
            kwargs["queries"] = queries
        with torch.inference_mode():
            try:
                kwargs["add_support_grid"] = True
                pred_tracks, _ = model(**kwargs)
            except TypeError:
                kwargs.pop("add_support_grid", None)
                pred_tracks, _ = model(**kwargs)

        last_processed_frame = chunk[-1][0]
        if is_first_step:
            is_first_step = False
            return
        if pred_tracks is None:
            return

        predicted = pred_tracks[0].detach().cpu().numpy()
        for local_index, (absolute_frame, _, original_rgb) in enumerate(chunk):
            if absolute_frame <= last_displayed_frame:
                continue
            predicted_index = absolute_frame - start_frame
            if predicted_index >= len(predicted):
                predicted_index = local_index
            points = predicted[predicted_index] / model_scale
            if freeze_lost:
                points, last_valid_points, visible_points = filter_visible_points(
                    points,
                    last_valid_points,
                    visible_points,
                    original_rgb,
                )
                grouped_tracks = regroup_visible_points(points, group_sizes, visible_points)
            else:
                grouped_tracks = regroup_points(points, group_sizes)
            emit_tracked_frame(
                original_rgb,
                grouped_tracks,
                labels,
                output_writer,
                show_live_preview,
                frame_placeholder,
            )
            status_placeholder.caption(
                f"CoTracker3 frame {absolute_frame} / {end_frame} on {device}"
            )
            if show_live_preview:
                sync_to_video_clock(start_time, absolute_frame - start_frame, fps)
            last_displayed_frame = absolute_frame

    try:
        for absolute_frame in range(start_frame, end_frame + 1):
            relative_frame = absolute_frame - start_frame
            if relative_frame != 0 and relative_frame % step == 0:
                process_window()

            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            model_rgb, current_scale = resize_for_model(frame_rgb, model_max_side)
            if queries is None:
                model_scale = current_scale
                query_points = flat_points * model_scale
                query_data = np.column_stack(
                    [np.zeros(len(query_points), dtype=np.float32), query_points]
                ).astype(np.float32)
                queries = torch.from_numpy(query_data)[None].to(device)
                output_writer = open_output_writer(saved_path, frame_rgb, fps)
                emit_tracked_frame(
                    frame_rgb,
                    tracks,
                    labels,
                    output_writer,
                    show_live_preview,
                    frame_placeholder,
                )
            window.append((absolute_frame, model_rgb, frame_rgb))
            if len(window) > step * 2:
                window = window[-step * 2 :]

        if window and window[-1][0] > last_processed_frame:
            process_window()
    finally:
        cap.release()
        if output_writer is not None:
            output_writer.release()

    return saved_path


def track_with_cotracker3_offline(
    path: str,
    start_frame: int,
    end_frame: int,
    tracks: list[np.ndarray],
    labels: list[str],
    fps: float,
    model_max_side: int,
    device_name: str,
    frame_placeholder,
    status_placeholder,
    output_path: str | Path | None = None,
    show_live_preview: bool = False,
    freeze_lost: bool = False,
    chunk_frames: int = 64,
) -> Path | None:
    import torch

    flat_points, group_sizes = flatten_tracks(tracks)
    if len(flat_points) == 0:
        st.error("No points to track.")
        return

    with st.spinner("Loading CoTracker3 Offline..."):
        model, device = load_cotracker3_offline(device_name)

    chunk_frames = max(2, int(chunk_frames))
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output_writer = None
    saved_path = Path(output_path) if output_path else None
    start_time = time.perf_counter()
    last_valid_points = flat_points.copy()
    current_query_points = flat_points.copy()
    visible_points = np.ones(len(flat_points), dtype=bool)
    frames_written = 0

    try:
        absolute_frame = start_frame
        while absolute_frame <= end_frame:
            original_frames: list[np.ndarray] = []
            model_frames: list[np.ndarray] = []
            model_scale = 1.0
            chunk_start = absolute_frame

            for _ in range(chunk_frames):
                if absolute_frame > end_frame:
                    break
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                model_rgb, current_scale = resize_for_model(frame_rgb, model_max_side)
                if not model_frames:
                    model_scale = current_scale
                    if output_writer is None:
                        output_writer = open_output_writer(output_path, frame_rgb, fps)
                model_frames.append(model_rgb)
                original_frames.append(frame_rgb)
                absolute_frame += 1

            if not model_frames:
                break

            query_points = current_query_points * model_scale
            query_data = np.column_stack(
                [np.zeros(len(query_points), dtype=np.float32), query_points]
            ).astype(np.float32)
            video_tensor = (
                torch.from_numpy(np.stack(model_frames))
                .to(device=device, dtype=torch.float32)
                .permute(0, 3, 1, 2)[None]
            )
            queries = torch.from_numpy(query_data)[None].to(device)

            status_placeholder.caption(
                f"Running CoTracker3 Offline frames {chunk_start}-{absolute_frame - 1} / {end_frame} on {device}"
            )
            try:
                with torch.inference_mode():
                    pred_tracks, pred_visibility = model(video_tensor, queries=queries)
            except torch.cuda.OutOfMemoryError as error:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CoTracker3 Offline ran out of GPU memory for this chunk. "
                    "Lower `CoTracker Offline chunk frames`, lower `Neural model max side`, "
                    "or choose CPU for this tracker.\n\n"
                    f"Original error: {error}"
                ) from error

            predicted = pred_tracks[0].detach().cpu().numpy() / model_scale
            visibility = None
            if pred_visibility is not None:
                visibility = pred_visibility[0].detach().cpu().numpy().astype(bool)
            del video_tensor, queries, pred_tracks, pred_visibility
            if device.type == "cuda":
                torch.cuda.empty_cache()

            for relative_frame, frame_rgb in enumerate(original_frames):
                points = predicted[relative_frame]
                if freeze_lost:
                    valid_mask = visibility[relative_frame] if visibility is not None else None
                    points, last_valid_points, visible_points = filter_visible_points(
                        points,
                        last_valid_points,
                        visible_points,
                        frame_rgb,
                        valid_mask,
                    )
                    grouped_tracks = regroup_visible_points(points, group_sizes, visible_points)
                else:
                    grouped_tracks = regroup_points(points, group_sizes)
                emit_tracked_frame(
                    frame_rgb,
                    grouped_tracks,
                    labels,
                    output_writer,
                    show_live_preview,
                    frame_placeholder,
                )
                rendered_frame = chunk_start + relative_frame
                status_placeholder.caption(
                    f"CoTracker3 Offline frame {rendered_frame} / {end_frame} on {device}"
                )
                if show_live_preview:
                    sync_to_video_clock(start_time, rendered_frame - start_frame, fps)
                frames_written += 1

            if freeze_lost:
                current_query_points = last_valid_points.copy()
            else:
                current_query_points = predicted[-1].astype(np.float32)
    finally:
        cap.release()
        if output_writer is not None:
            output_writer.release()

    if frames_written == 0:
        st.error("Could not read frames for CoTracker3 Offline.")
        return
    return saved_path


def track_with_litetracker(
    path: str,
    start_frame: int,
    end_frame: int,
    tracks: list[np.ndarray],
    labels: list[str],
    fps: float,
    model_max_side: int,
    weights_path: str,
    device_name: str,
    frame_placeholder,
    status_placeholder,
    output_path: str | Path | None = None,
    show_live_preview: bool = False,
    freeze_lost: bool = False,
) -> Path | None:
    import torch

    flat_points, group_sizes = flatten_tracks(tracks)
    if len(flat_points) == 0:
        st.error("No points to track.")
        return

    with st.spinner("Loading LiteTracker..."):
        model, device = load_litetracker(weights_path, device_name)
    if hasattr(model, "init_video_online_processing"):
        model.init_video_online_processing()

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    queries = None
    start_time = time.perf_counter()
    output_writer = None
    saved_path = Path(output_path) if output_path else None
    last_valid_points = flat_points.copy()
    visible_points = np.ones(len(flat_points), dtype=bool)

    try:
        with torch.inference_mode():
            for absolute_frame in range(start_frame, end_frame + 1):
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                model_rgb, model_scale = resize_for_model(frame_rgb, model_max_side)
                if queries is None:
                    query_points = flat_points * model_scale
                    query_data = np.column_stack(
                        [np.zeros(len(query_points), dtype=np.float32), query_points]
                    ).astype(np.float32)
                    queries = torch.from_numpy(query_data)[None].to(device)
                    output_writer = open_output_writer(saved_path, frame_rgb, fps)

                frame_tensor = (
                    torch.from_numpy(model_rgb)
                    .to(device=device, dtype=torch.float32)
                    .permute(2, 0, 1)[None]
                )
                coords, _, _ = model(frame_tensor, queries=queries)
                points = coords[0, -1].detach().cpu().numpy() / model_scale
                if freeze_lost:
                    points, last_valid_points, visible_points = filter_visible_points(
                        points,
                        last_valid_points,
                        visible_points,
                        frame_rgb,
                    )
                    grouped_tracks = regroup_visible_points(points, group_sizes, visible_points)
                else:
                    grouped_tracks = regroup_points(points, group_sizes)
                emit_tracked_frame(
                    frame_rgb,
                    grouped_tracks,
                    labels,
                    output_writer,
                    show_live_preview,
                    frame_placeholder,
                )
                status_placeholder.caption(
                    f"LiteTracker frame {absolute_frame} / {end_frame} on {device}"
                )
                if show_live_preview:
                    sync_to_video_clock(start_time, absolute_frame - start_frame, fps)
    finally:
        cap.release()
        if output_writer is not None:
            output_writer.release()

    return saved_path


def track_with_lk(
    path: str,
    start_frame: int,
    end_frame: int,
    tracks: list[np.ndarray],
    labels: list[str],
    frame_skip: int,
    fps: float,
    frame_placeholder,
    status_placeholder,
    output_path: str | Path | None = None,
    show_live_preview: bool = False,
    freeze_lost: bool = False,
) -> Path | None:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, previous_bgr = cap.read()
    if not ok:
        st.error("Could not read the selected start frame.")
        cap.release()
        return

    previous_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
    active_tracks = [track.copy().astype(np.float32) for track in tracks]
    visible_tracks = [np.ones(len(track), dtype=bool) for track in active_tracks]
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    current_frame = start_frame
    start_time = time.perf_counter()
    fps = max(fps, 1.0)
    rgb = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2RGB)
    output_writer = open_output_writer(output_path, rgb, fps)
    saved_path = Path(output_path) if output_path else None
    displayed_tracks = [
        points[visible_tracks[index]]
        for index, points in enumerate(active_tracks)
    ] if freeze_lost else active_tracks
    emit_tracked_frame(rgb, displayed_tracks, labels, output_writer, show_live_preview, frame_placeholder)

    try:
        while current_frame < end_frame:
            target_frame = min(current_frame + frame_skip, end_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ok, next_bgr = cap.read()
            if not ok:
                break

            next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
            next_rgb = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2RGB)
            for index, points in enumerate(active_tracks):
                if len(points) == 0:
                    continue
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    previous_gray,
                    next_gray,
                    points.reshape(-1, 1, 2),
                    None,
                    **lk_params,
                )
                if next_points is not None and status is not None:
                    status = status.reshape(-1).astype(bool)
                    updated = points.copy()
                    candidate_points = next_points.reshape(-1, 2)
                    if freeze_lost:
                        updated, _, visible_tracks[index] = filter_visible_points(
                            candidate_points,
                            updated,
                            visible_tracks[index],
                            next_rgb,
                            status,
                        )
                    else:
                        updated[status] = candidate_points[status]
                    active_tracks[index] = updated
                elif freeze_lost:
                    visible_tracks[index][:] = False

            current_frame = target_frame
            previous_gray = next_gray
            rgb = next_rgb
            displayed_tracks = [
                points[visible_tracks[index]]
                for index, points in enumerate(active_tracks)
            ] if freeze_lost else active_tracks
            emit_tracked_frame(
                rgb,
                displayed_tracks,
                labels,
                output_writer,
                show_live_preview,
                frame_placeholder,
            )
            status_placeholder.caption(f"Tracking frame {current_frame} / {end_frame}")
            if show_live_preview:
                sync_to_video_clock(start_time, current_frame - start_frame, fps)
    finally:
        cap.release()
        if output_writer is not None:
            output_writer.release()

    return saved_path


def run_external_tracker(
    tracker_name: str,
    command_template: str,
    video_path: str | Path,
    start_frame: int,
    tracks: list[np.ndarray],
    labels: list[str],
    output_path: str | Path,
    device_name: str,
    status_placeholder,
    freeze_lost: bool = False,
) -> Path:
    import subprocess

    if not command_template.strip():
        raise RuntimeError(f"No command configured for {tracker_name}.")

    output_path = Path(output_path)
    prompt_path = output_path.with_suffix(".prompts.json")
    device = "cuda" if device_name == CUDA_DEVICE else "cpu"
    command = command_template.format(
        video=shlex.quote(str(video_path)),
        start_frame=int(start_frame),
        prompts=shlex.quote(str(prompt_path)),
        output=shlex.quote(str(output_path)),
        device=device,
        freeze_lost_points="--hide-lost-points" if freeze_lost else "",
    )
    if freeze_lost and "--hide-lost-points" not in command and "--freeze-lost-points" not in command:
        command = f"{command} --hide-lost-points"
    command_parts = shlex.split(command)
    if not command_parts:
        raise RuntimeError(f"No command configured for {tracker_name}.")
    executable = command_parts[0]
    executable_path = Path(executable).expanduser()
    if (executable_path.is_absolute() or "/" in executable) and not executable_path.exists():
        raise RuntimeError(
            f"{tracker_name} command executable does not exist: {executable}\n\n"
            f"{external_tracker_setup_instructions(tracker_name)}"
        )
    if "/" not in executable and shutil.which(executable) is None:
        raise RuntimeError(
            f"{tracker_name} command executable was not found on PATH: {executable}\n\n"
            f"{external_tracker_setup_instructions(tracker_name)}"
        )
    if Path(executable).name.startswith("python") and len(command_parts) > 1:
        script_path = Path(command_parts[1]).expanduser()
        if not script_path.exists():
            raise RuntimeError(
                f"{tracker_name} inference script does not exist: {script_path}\n\n"
                f"{external_tracker_setup_instructions(tracker_name)}"
            )

    write_prompt_file(prompt_path, tracks, labels, start_frame)
    status_placeholder.caption(f"Running {tracker_name} external command...")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "No error output."
        raise RuntimeError(
            f"{tracker_name} command failed.\n\nCommand:\n{command}\n\nOutput:\n{message}"
        )
    if not output_path.exists():
        raise RuntimeError(
            f"{tracker_name} command finished but did not create the expected output: {output_path}"
        )
    return output_path
