from __future__ import annotations

import csv
import hashlib
import importlib
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import tracking_methods as _tracking_methods

importlib.reload(_tracking_methods)
from tracking_methods import (
    COTRACKER_OFFLINE_TRACKER,
    COTRACKER_TRACKER,
    CPU_DEVICE,
    CUDA_DEVICE,
    InstrumentAvoidanceConfig,
    LITETRACKER_TRACKER,
    MEDSAM2_TRACKER,
    OPENCV_TRACKER,
    SAM2_TRACKER,
    SAM3_TRACKER,
    SURGISAM2_TRACKER,
    TrackValidationConfig,
    create_comparison_collage,
    cuda_is_available,
    cuda_summary,
    default_external_command,
    default_litetracker_weights_path,
    download_litetracker_weights,
    draw_tracks,
    external_tracker_is_available,
    external_tracker_setup_instructions,
    run_external_tracker,
    track_with_cotracker3_offline,
    track_with_cotracker3_online,
    track_with_litetracker,
    track_with_lk,
    tracker_slug,
    unavailable_external_tracker_message,
)


def install_drawable_canvas_streamlit_compat() -> None:
    """Keep streamlit-drawable-canvas working with newer Streamlit releases."""
    import streamlit.elements.image as st_image

    if hasattr(st_image, "image_to_url"):
        return

    from streamlit.elements.lib.image_utils import image_to_url as current_image_to_url
    from streamlit.elements.lib.layout_utils import create_layout_config

    def image_to_url(image, width, clamp, channels, output_format, image_id):
        layout_config = create_layout_config(width=width)
        return current_image_to_url(
            image,
            layout_config,
            clamp,
            channels,
            output_format,
            image_id,
        )

    st_image.image_to_url = image_to_url


install_drawable_canvas_streamlit_compat()
from streamlit_drawable_canvas import st_canvas


TEMP_DIR = Path(tempfile.gettempdir()) / "surgical_video_tracker"
UPLOAD_DIR = TEMP_DIR / "uploads"
OUTPUT_DIR = TEMP_DIR / "output"
MAX_CANVAS_WIDTH = 900
TRACKER_OPTIONS = [
    OPENCV_TRACKER,
    COTRACKER_TRACKER,
    COTRACKER_OFFLINE_TRACKER,
    LITETRACKER_TRACKER,
    SAM2_TRACKER,
    SURGISAM2_TRACKER,
    SAM3_TRACKER,
    MEDSAM2_TRACKER,
]


def parse_obj_model(obj_bytes: bytes) -> tuple[np.ndarray, list[list[int]]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    text = obj_bytes.decode("utf-8", errors="ignore")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            try:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
        elif parts[0] == "f" and len(parts) >= 4:
            face: list[int] = []
            for token in parts[1:]:
                index_text = token.split("/")[0]
                if not index_text:
                    continue
                try:
                    index = int(index_text)
                except ValueError:
                    continue
                if index < 0:
                    index = len(vertices) + index
                else:
                    index -= 1
                if index >= 0:
                    face.append(index)
            if len(face) >= 3:
                faces.append(face)
    if not vertices:
        raise RuntimeError("The OBJ file does not contain any vertices.")
    return np.asarray(vertices, dtype=np.float32), faces


def normalize_obj_vertices(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(centered, axis=1).max())
    if radius <= 0.0:
        return centered
    return centered / radius


def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = (math.radians(value) for value in (rx_deg, ry_deg, rz_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def project_obj_vertices(
    vertices: np.ndarray,
    frame_width: int,
    frame_height: int,
    center_x: float,
    center_y: float,
    scale_px: float,
    rotate_x: float,
    rotate_y: float,
    rotate_z: float,
) -> np.ndarray:
    normalized = normalize_obj_vertices(vertices)
    rotated = normalized @ rotation_matrix_xyz(rotate_x, rotate_y, rotate_z).T
    projected = np.empty((len(rotated), 2), dtype=np.float32)
    projected[:, 0] = float(center_x) + rotated[:, 0] * float(scale_px)
    projected[:, 1] = float(center_y) - rotated[:, 1] * float(scale_px)
    projected[:, 0] = np.clip(projected[:, 0], -frame_width, frame_width * 2)
    projected[:, 1] = np.clip(projected[:, 1], -frame_height, frame_height * 2)
    return projected


def draw_obj_overlay(
    frame_rgb: np.ndarray,
    points_xy: np.ndarray,
    faces: list[list[int]],
    color: tuple[int, int, int] = (60, 220, 255),
) -> np.ndarray:
    output = frame_rgb.copy()
    if not len(points_xy):
        return output
    for face in faces[:8000]:
        valid_face = [index for index in face if 0 <= index < len(points_xy)]
        if len(valid_face) < 2:
            continue
        pts = np.round(points_xy[valid_face]).astype(np.int32)
        cv2.polylines(output, [pts], True, color, 1, lineType=cv2.LINE_AA)
    for point in np.round(points_xy[:: max(1, len(points_xy) // 120)]).astype(np.int32):
        cv2.circle(output, tuple(point), 2, color, -1, lineType=cv2.LINE_AA)
    return output


def obj_tracks_from_projection(
    points_xy: np.ndarray,
    frame_width: int,
    frame_height: int,
    max_points: int,
) -> tuple[list[np.ndarray], list[str]]:
    if max_points <= 0 or len(points_xy) == 0:
        return [], []
    in_frame = (
        (points_xy[:, 0] >= 0)
        & (points_xy[:, 0] < frame_width)
        & (points_xy[:, 1] >= 0)
        & (points_xy[:, 1] < frame_height)
    )
    visible_points = points_xy[in_frame]
    if len(visible_points) == 0:
        return [], []
    step = max(1, int(math.ceil(len(visible_points) / max_points)))
    selected = visible_points[::step][:max_points].astype(np.float32)
    tracks = [point.reshape(1, 2) for point in selected]
    labels = [f"obj {index}" for index in range(1, len(tracks) + 1)]
    return tracks, labels


def obj_control_box_drawing(width: int, height: int) -> dict[str, Any]:
    box_size = max(60, min(width, height) // 3)
    return {
        "version": "4.4.0",
        "objects": [
            {
                "type": "rect",
                "left": (width - box_size) / 2,
                "top": (height - box_size) / 2,
                "width": box_size,
                "height": box_size,
                "scaleX": 1,
                "scaleY": 1,
                "fill": "rgba(60, 220, 255, 0.08)",
                "stroke": "#3cdcff",
                "strokeWidth": 3,
                "transparentCorners": False,
                "cornerColor": "#3cdcff",
            }
        ],
    }


def obj_placement_from_canvas(
    json_data: dict[str, Any] | None,
    canvas_scale: float,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float]:
    default_scale = max(40.0, min(frame_width, frame_height) / 4.0)
    if not json_data:
        return frame_width / 2.0, frame_height / 2.0, default_scale
    for obj in json_data.get("objects", []):
        if obj.get("type") != "rect":
            continue
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))
        width = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0))
        height = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0))
        center_x = (left + width / 2.0) / canvas_scale
        center_y = (top + height / 2.0) / canvas_scale
        scale_px = max(10.0, max(width, height) / (2.0 * canvas_scale))
        return center_x, center_y, scale_px
    return frame_width / 2.0, frame_height / 2.0, default_scale


def cleanup_path(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def cleanup_paths(paths: list[Path]) -> None:
    unique_paths = {path for path in paths if path}
    for path in sorted(unique_paths, key=lambda item: len(item.parts), reverse=True):
        cleanup_path(path)


def cleanup_empty_work_dirs() -> None:
    for path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
        try:
            path.rmdir()
        except OSError:
            pass


def reset_work_dir() -> None:
    cleanup_path(TEMP_DIR)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_video(uploaded_file) -> Path:
    reset_work_dir()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    hasher = hashlib.sha1()
    tmp_path = UPLOAD_DIR / f".upload-{time.time_ns()}{suffix}"

    try:
        uploaded_file.seek(0)
        with tmp_path.open("wb") as output:
            while chunk := uploaded_file.read(1024 * 1024):
                hasher.update(chunk)
                output.write(chunk)
        uploaded_file.seek(0)
    except OSError as error:
        tmp_path.unlink(missing_ok=True)
        cleanup_empty_work_dirs()
        raise RuntimeError(
            "Not enough free disk space to prepare this upload. "
            "Choose a shorter/smaller video, free disk space, or upload an already-trimmed clip."
        ) from error

    path = UPLOAD_DIR / f"{hasher.hexdigest()[:12]}{suffix}"
    tmp_path.replace(path)
    return path


def create_test_output_dir(video_path: Path, start_frame: int, mode: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{video_path.stem}_{mode}_f{start_frame}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def output_video_path(test_dir: Path, tracker_name: str) -> Path:
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir / f"{tracker_slug(tracker_name)}.mp4"


def comparison_video_path(test_dir: Path) -> Path:
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir / "comparison.mp4"


def make_streamlit_preview_video(path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return path

    preview_path = path.with_name(f"{path.stem}_preview.mp4")
    tmp_path = preview_path.with_suffix(".tmp.mp4")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        return path
    tmp_path.replace(preview_path)
    return preview_path


def timing_log_path(test_dir: Path) -> Path:
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir / "model_timings.csv"


def append_timing_log(test_dir: Path, row: dict[str, object]) -> Path:
    path = timing_log_path(test_dir)
    fieldnames = [
        "timestamp",
        "video",
        "mode",
        "model",
        "device",
        "start_frame",
        "end_frame",
        "annotation_count",
        "point_count",
        "status",
        "duration_seconds",
        "output_path",
        "error",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def can_read_first_frame(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    return bool(ok and frame is not None)


def make_opencv_compatible_video(path: Path) -> Path:
    compatible_path = path.with_name(f"{path.stem}_opencv.mp4")
    if compatible_path.exists() and can_read_first_frame(compatible_path):
        return compatible_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "OpenCV could not decode this video and ffmpeg is not available. "
            "Try converting the video to an OpenCV-friendly MP4 first."
        )

    tmp_path = compatible_path.with_suffix(".tmp.mp4")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "mpeg4",
        "-q:v",
        "3",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not convert video for OpenCV:\n{result.stderr.strip()}")

    tmp_path.replace(compatible_path)
    if not can_read_first_frame(compatible_path):
        raise RuntimeError("The converted video still could not be decoded by OpenCV.")
    return compatible_path


def prepare_video_for_opencv(path: Path) -> Path:
    if can_read_first_frame(path):
        return path
    return make_opencv_compatible_video(path)


def trim_video_interval(path: Path, start_frame: int, end_frame: int, fps: float) -> Path:
    clip_path = path.with_name(f"{path.stem}_clip_f{start_frame}_{end_frame}.mp4")
    if clip_path.exists() and can_read_first_frame(clip_path):
        return clip_path

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video to create the selected interval.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 25.0)
    tmp_path = clip_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the selected interval video.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    while current_frame <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        current_frame += 1

    cap.release()
    writer.release()

    if current_frame <= start_frame:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Could not read any frames for the selected interval.")

    tmp_path.replace(clip_path)
    return clip_path


@st.cache_data(show_spinner=False)
def video_info(path: str) -> dict[str, float | int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"frame_count": frame_count, "fps": fps, "width": width, "height": height}


@st.cache_data(show_spinner=False)
def read_frame(path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}.")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def format_timecode(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_timecode(value: str) -> int:
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3 or any(part.strip() == "" for part in parts):
        raise ValueError("Use hh:mm:ss, mm:ss, or seconds.")
    try:
        numbers = [int(part) for part in parts]
    except ValueError as error:
        raise ValueError("Time values must contain only numbers and `:`.") from error
    if any(number < 0 for number in numbers):
        raise ValueError("Time values cannot be negative.")
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        minutes, seconds = numbers
        if seconds >= 60:
            raise ValueError("Seconds must be less than 60.")
        return minutes * 60 + seconds
    hours, minutes, seconds = numbers
    if minutes >= 60 or seconds >= 60:
        raise ValueError("Minutes and seconds must be less than 60.")
    return hours * 3600 + minutes * 60 + seconds


def resize_for_canvas(frame_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = frame_rgb.shape[:2]
    scale = min(1.0, MAX_CANVAS_WIDTH / width)
    if scale == 1.0:
        return frame_rgb, scale
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame_rgb, new_size, interpolation=cv2.INTER_AREA), scale


def fabric_points(obj: dict[str, Any]) -> list[tuple[float, float]]:
    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    scale_x = float(obj.get("scaleX", 1))
    scale_y = float(obj.get("scaleY", 1))
    width = float(obj.get("width", 0))
    height = float(obj.get("height", 0))
    obj_type = obj.get("type")

    if obj_type == "circle":
        radius = float(obj.get("radius", 0))
        return [(left + radius * scale_x, top + radius * scale_y)]

    if obj_type == "rect":
        width *= scale_x
        height *= scale_y
        return [
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
        ]

    if obj_type in {"path", "polygon"}:
        raw_points: list[tuple[float, float]] = []
        for point in obj.get("points", []):
            if "x" in point and "y" in point:
                raw_points.append((float(point["x"]), float(point["y"])))
        for entry in obj.get("path", []):
            if len(entry) >= 3 and entry[0] in {"M", "L"}:
                raw_points.append((float(entry[1]), float(entry[2])))

        if not raw_points:
            return []

        path_offset = obj.get("pathOffset") or {}
        if "x" in path_offset and "y" in path_offset:
            offset_x = float(path_offset["x"])
            offset_y = float(path_offset["y"])
            origin_x = left + (width * scale_x) / 2
            origin_y = top + (height * scale_y) / 2
            return [
                (origin_x + (x - offset_x) * scale_x, origin_y + (y - offset_y) * scale_y)
                for x, y in raw_points
            ]

        xs = [point[0] for point in raw_points]
        ys = [point[1] for point in raw_points]
        looks_local = (
            min(xs) >= -2
            and min(ys) >= -2
            and max(xs) <= width + 2
            and max(ys) <= height + 2
        )
        if looks_local:
            return [(left + x * scale_x, top + y * scale_y) for x, y in raw_points]
        return [(x * scale_x, y * scale_y) for x, y in raw_points]

    return []


def parse_annotations(
    json_data: dict[str, Any] | None,
    canvas_scale: float,
) -> tuple[list[np.ndarray], list[str]]:
    tracks: list[np.ndarray] = []
    labels: list[str] = []
    if not json_data:
        return tracks, labels

    objects = json_data.get("objects", [])
    for index, obj in enumerate(objects, start=1):
        points = fabric_points(obj)
        if not points:
            continue
        original_points = np.array(points, dtype=np.float32) / canvas_scale
        tracks.append(original_points)
        labels.append(f"{obj.get('type', 'shape')} {index}")
    return tracks, labels


def parse_annotation_regions(
    json_data: dict[str, Any] | None,
    canvas_scale: float,
) -> list[np.ndarray]:
    regions: list[np.ndarray] = []
    if not json_data:
        return regions

    for obj in json_data.get("objects", []):
        if obj.get("type") not in {"rect", "path", "polygon"}:
            continue
        points = fabric_points(obj)
        if len(points) < 3:
            continue
        regions.append((np.array(points, dtype=np.float32) / canvas_scale).astype(np.float32))
    return regions


def generate_grid_tracks(
    frame_width: int,
    frame_height: int,
    spacing: int,
    margin: int,
    max_points: int,
    regions: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    x_values = np.arange(margin, max(margin + 1, frame_width - margin), spacing, dtype=np.float32)
    y_values = np.arange(margin, max(margin + 1, frame_height - margin), spacing, dtype=np.float32)
    points: list[np.ndarray] = []
    for y in y_values:
        for x in x_values:
            if regions and not any(cv2.pointPolygonTest(region, (float(x), float(y)), False) >= 0 for region in regions):
                continue
            points.append(np.array([[x, y]], dtype=np.float32))
    if max_points > 0:
        center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)
        points.sort(key=lambda item: float(np.linalg.norm(item[0] - center)))
        points = points[:max_points]
    labels = [f"grid {index}" for index in range(1, len(points) + 1)]
    return points, labels


LOCAL_NEURAL_TRACKERS = {
    COTRACKER_TRACKER,
    COTRACKER_OFFLINE_TRACKER,
    LITETRACKER_TRACKER,
}


def tracker_device_for_run(
    tracker_name: str,
    requested_device: str,
    enable_gpu_local_neural: bool,
) -> str:
    if tracker_name in LOCAL_NEURAL_TRACKERS and not enable_gpu_local_neural:
        return CPU_DEVICE
    return requested_device


def format_tracker_error(error: RuntimeError) -> str:
    message = str(error)
    cuda_markers = [
        "CUDA error",
        "cudaErrorLaunchFailure",
        "unspecified launch failure",
        "CUDA out of memory",
    ]
    if any(marker in message for marker in cuda_markers):
        return (
            f"{message}\n\n"
            "CUDA is now poisoned in this Python process. Restart Streamlit/Python before any CUDA test. "
            "For a safer retry, disable `Use GPU for CoTracker/LiteTracker` or reduce points/interval/model size."
        )
    return message


def run_tracker_model(
    tracker_name: str,
    video_path: Path,
    start_frame: int,
    end_frame: int,
    tracks: list[np.ndarray],
    labels: list[str],
    fps: float,
    frame_skip: int,
    model_max_side: int,
    lite_weights_path: str,
    cotracker_offline_chunk_frames: int,
    device_name: str,
    external_commands: dict[str, str],
    result_path: Path,
    frame_placeholder,
    status_placeholder,
    show_live_preview: bool,
    freeze_lost_points: bool,
    instrument_avoidance: InstrumentAvoidanceConfig | None,
    track_validation: TrackValidationConfig | None,
) -> Path | None:
    if tracker_name == OPENCV_TRACKER:
        return track_with_lk(
            str(video_path),
            start_frame,
            end_frame,
            tracks,
            labels,
            frame_skip,
            fps,
            frame_placeholder,
            status_placeholder,
            result_path,
            show_live_preview,
            freeze_lost_points,
            instrument_avoidance,
            track_validation,
        )
    if tracker_name == COTRACKER_TRACKER:
        return track_with_cotracker3_online(
            str(video_path),
            start_frame,
            end_frame,
            tracks,
            labels,
            fps,
            model_max_side,
            device_name,
            frame_placeholder,
            status_placeholder,
            result_path,
            show_live_preview,
            freeze_lost_points,
            instrument_avoidance,
            track_validation,
        )
    if tracker_name == COTRACKER_OFFLINE_TRACKER:
        return track_with_cotracker3_offline(
            str(video_path),
            start_frame,
            end_frame,
            tracks,
            labels,
            fps,
            model_max_side,
            device_name,
            frame_placeholder,
            status_placeholder,
            result_path,
            show_live_preview,
            freeze_lost_points,
            cotracker_offline_chunk_frames,
            instrument_avoidance,
            track_validation,
        )
    if tracker_name == LITETRACKER_TRACKER:
        return track_with_litetracker(
            str(video_path),
            start_frame,
            end_frame,
            tracks,
            labels,
            fps,
            model_max_side,
            lite_weights_path,
            device_name,
            frame_placeholder,
            status_placeholder,
            result_path,
            show_live_preview,
            freeze_lost_points,
            instrument_avoidance,
            track_validation,
        )
    if tracker_name in {SAM2_TRACKER, SURGISAM2_TRACKER, SAM3_TRACKER, MEDSAM2_TRACKER}:
        return run_external_tracker(
            tracker_name,
            external_commands.get(tracker_name, default_external_command(tracker_name)),
            video_path,
            start_frame,
            tracks,
            labels,
            result_path,
            device_name,
            status_placeholder,
            freeze_lost_points,
        )
    raise RuntimeError(f"Unknown tracker: {tracker_name}")


st.set_page_config(page_title="Surgical Video Tracker", layout="wide")
st.title("Surgical Video Tracker")

if st.session_state.get("model_cache_version") != "instrument_avoidance_v1":
    st.cache_resource.clear()
    st.session_state["model_cache_version"] = "instrument_avoidance_v1"

with st.sidebar:
    uploaded = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])
    uploaded_obj = st.file_uploader("3D model overlay (.obj)", type=["obj"])
    mode = st.radio("Annotation", ["point", "rect", "polygon"], horizontal=True)
    stroke_width = st.slider("Line width", 1, 8, 3)
    stroke_color = st.color_picker("Color", "#ff485c")
    add_point_cloud = st.checkbox("Add point cloud", value=False)
    point_cloud_area = "Drawn rect/polygon areas"
    grid_spacing = 120
    grid_margin = 48
    grid_max_points = 50
    if add_point_cloud:
        point_cloud_area = st.selectbox(
            "Point cloud area",
            ["Drawn rect/polygon areas", "Full frame"],
        )
        point_cloud_preset = st.selectbox("Point cloud speed", ["Fast", "Balanced", "Dense"])
        preset_spacing, preset_max_points = {
            "Fast": (120, 50),
            "Balanced": (96, 100),
            "Dense": (64, 200),
        }[point_cloud_preset]
        grid_spacing = st.slider("Grid spacing", 8, 200, preset_spacing, 8)
        grid_margin = st.slider("Grid margin", 0, 160, 48, 4)
        grid_max_points = st.slider("Max grid points", 10, 500, preset_max_points, 10)
    enable_gpu_local_neural = st.checkbox("Use GPU for CoTracker/LiteTracker", value=False)
    st.divider()
    unavailable_trackers = [
        tracker
        for tracker in TRACKER_OPTIONS
        if tracker in {SAM2_TRACKER, SURGISAM2_TRACKER, SAM3_TRACKER, MEDSAM2_TRACKER}
        and not external_tracker_is_available(tracker)
    ]
    tracker_options = [tracker for tracker in TRACKER_OPTIONS if tracker not in unavailable_trackers]
    if unavailable_trackers:
        with st.expander("Unavailable external trackers", expanded=False):
            for unavailable_tracker in unavailable_trackers:
                st.warning(unavailable_external_tracker_message(unavailable_tracker))
    compare_mode = st.checkbox("Compare models", value=False)
    if compare_mode:
        selected_trackers = st.multiselect(
            "Models to compare",
            tracker_options,
            default=[OPENCV_TRACKER, COTRACKER_TRACKER],
        )
        tracker_name = selected_trackers[0] if selected_trackers else OPENCV_TRACKER
        collage_tile_width = st.slider("Collage tile width", 320, 960, 640, 64)
    else:
        tracker_name = st.selectbox("Tracker", tracker_options)
        selected_trackers = [tracker_name]
        collage_tile_width = 640
    frame_skip = st.slider("OpenCV frame step", 1, 10, 1)
    model_max_side = st.slider("Neural model max side", 256, 1024, 384, 64)
    freeze_lost_points = st.checkbox("Hide lost points and resume when visible", value=True)
    reject_drift_points = st.checkbox("Reject border/jump drift", value=True)
    track_validation: TrackValidationConfig | None = None
    if reject_drift_points:
        edge_margin_px = st.slider("Reject points within edge px", 0, 120, 32, 1)
        max_jump_px = st.slider("Reject point jumps over px", 0, 300, 50, 5)
        content_margin_px = st.slider("Reject points near content edge px", 0, 200, 48, 2)
        track_validation = TrackValidationConfig(
            edge_margin=edge_margin_px,
            max_jump_px=float(max_jump_px),
            content_margin=content_margin_px,
        )
        st.caption("Hides points that stick to frame/content borders or jump too far between frames.")
    default_instrument_onnx_path = Path("instrument_segmentation/runs/instrument_model/best.onnx")
    avoid_instruments = st.checkbox(
        "Avoid instruments with ONNX mask",
        value=default_instrument_onnx_path.exists(),
    )
    instrument_avoidance: InstrumentAvoidanceConfig | None = None
    if avoid_instruments:
        instrument_onnx_path = st.text_input(
            "Instrument ONNX model",
            str(default_instrument_onnx_path),
        )
        instrument_mask_size = st.slider("Instrument mask model size", 128, 1024, 512, 64)
        instrument_threshold = st.slider("Instrument mask threshold", 0.05, 0.95, 0.35, 0.05)
        instrument_dilation = st.slider("Instrument avoid margin px", 0, 60, 15, 1)
        instrument_avoidance = InstrumentAvoidanceConfig(
            onnx_path=instrument_onnx_path,
            image_size=instrument_mask_size,
            threshold=instrument_threshold,
            dilation=instrument_dilation,
        )
        st.caption(
            "Applies to OpenCV, CoTracker, and LiteTracker. Points on the instrument mask are hidden and resume when they leave it."
        )
    show_live_preview = st.checkbox("Live preview while tracking/collage", value=False)
    st.caption("Uploads and outputs are temporary. Use the download buttons to keep results.")

    device_name = CPU_DEVICE
    uses_external_device = any(
        tracker
        in {
            SAM2_TRACKER,
            SURGISAM2_TRACKER,
            SAM3_TRACKER,
            MEDSAM2_TRACKER,
        }
        for tracker in selected_trackers
    )
    uses_local_neural = any(tracker in LOCAL_NEURAL_TRACKERS for tracker in selected_trackers)
    if uses_external_device or (uses_local_neural and enable_gpu_local_neural):
        device_options = [CUDA_DEVICE, CPU_DEVICE] if cuda_is_available() else [CPU_DEVICE, CUDA_DEVICE]
        device_name = st.selectbox("Neural device", device_options, index=0)
        gpu_message = cuda_summary()
        if device_name == CUDA_DEVICE and "not visible" in gpu_message:
            st.warning(gpu_message)
        else:
            st.caption(gpu_message)
        if device_name == CUDA_DEVICE and add_point_cloud:
            st.warning(
                "Dense point clouds can exceed GPU memory or trigger CUDA launch failures. "
                "Use a drawn area, larger spacing, fewer max points, or CPU if this happens."
            )
        if uses_local_neural and enable_gpu_local_neural and device_name == CUDA_DEVICE:
            st.warning(
                "CoTracker/LiteTracker will run on GPU. If CUDA fails, restart Streamlit before trying again."
            )
    elif uses_local_neural:
        st.caption("CoTracker/LiteTracker will use CPU unless `Use GPU for CoTracker/LiteTracker` is enabled.")

    lite_weights_path = str(default_litetracker_weights_path())
    if LITETRACKER_TRACKER in selected_trackers:
        lite_weights_path = st.text_input("LiteTracker weights", lite_weights_path)
        if st.button("Download LiteTracker weights"):
            with st.spinner("Downloading CoTracker3 scaled online weights..."):
                download_litetracker_weights(Path(lite_weights_path).expanduser())
            st.success("Weights downloaded.")

    cotracker_offline_chunk_frames = 64
    if COTRACKER_OFFLINE_TRACKER in selected_trackers:
        cotracker_offline_chunk_frames = st.slider(
            "CoTracker Offline chunk frames",
            8,
            240,
            64,
            8,
        )

    external_commands: dict[str, str] = {}
    for external_tracker in [SAM2_TRACKER, SURGISAM2_TRACKER, SAM3_TRACKER, MEDSAM2_TRACKER]:
        if external_tracker in selected_trackers:
            command_key = f"external_command_{external_tracker}"
            if external_tracker == SAM3_TRACKER:
                command_key = "external_command_SAM3_track_env"
            external_commands[external_tracker] = st.text_area(
                f"{external_tracker} command",
                default_external_command(external_tracker),
                height=110,
                key=command_key,
            )
            st.caption("Available placeholders: `{video}`, `{start_frame}`, `{prompts}`, `{output}`, `{device}`.")
    st.caption("Tracking runs inside the selected interval at normal video speed.")

if not uploaded:
    st.info("Upload a surgical video to choose a frame and mark points or regions.")
    st.stop()

try:
    original_video_path = save_uploaded_video(uploaded)
except RuntimeError as error:
    st.error(str(error))
    st.stop()
transient_input_paths = [original_video_path]
try:
    with st.spinner("Preparing video for OpenCV..."):
        video_path = prepare_video_for_opencv(original_video_path)
except RuntimeError as error:
    cleanup_paths(transient_input_paths)
    st.error(str(error))
    st.stop()

if video_path != original_video_path:
    transient_input_paths.append(video_path)
    st.info("This video needed an OpenCV-compatible temporary copy.")

try:
    info = video_info(str(video_path))
    frame_count = int(info["frame_count"])
    fps = float(info["fps"])
    last_frame = max(0, frame_count - 1)
    duration_seconds = last_frame / fps if fps else float(last_frame)

    use_clip_interval = st.checkbox("Use only a selected video interval", value=False)
    if use_clip_interval and frame_count > 1:
        start_col, end_col = st.columns(2)
        with start_col:
            clip_start_text = st.text_input(
                "Interval start",
                "00:00:00",
                help="Use hh:mm:ss, mm:ss, or seconds.",
            )
        with end_col:
            clip_end_text = st.text_input(
                "Interval end",
                format_timecode(duration_seconds),
                help="Use hh:mm:ss, mm:ss, or seconds.",
            )
        try:
            clip_start_second = parse_timecode(clip_start_text)
            clip_end_second = parse_timecode(clip_end_text)
        except ValueError as error:
            st.error(str(error))
            st.stop()
        max_second = int(round(duration_seconds))
        if clip_start_second > max_second:
            st.error(f"Interval start is after the video duration ({format_timecode(duration_seconds)}).")
            st.stop()
        if clip_end_second <= clip_start_second:
            st.error("Interval end must be after interval start.")
            st.stop()
        clip_end_second = min(clip_end_second, max_second)
        clip_start_frame = min(last_frame, max(0, int(round(clip_start_second * fps))))
        clip_end_frame = min(last_frame, max(clip_start_frame, int(round(clip_end_second * fps))))
    else:
        clip_start_frame, clip_end_frame = 0, last_frame

    clip_start_time = clip_start_frame / fps if fps else 0.0
    clip_end_time = clip_end_frame / fps if fps else 0.0
    clip_duration = max(0.0, (clip_end_frame - clip_start_frame + 1) / fps) if fps else 0.0
    st.caption(
        "Selected interval: "
        f"frames {clip_start_frame}-{clip_end_frame} "
        f"({format_timecode(clip_start_time)} to {format_timecode(clip_end_time)}, "
        f"{format_timecode(clip_duration)} total)."
    )

    previous_frame_index = int(st.session_state.get("frame_index", clip_start_frame))
    previous_frame_index = min(max(previous_frame_index, clip_start_frame), clip_end_frame)
    frame_index = st.slider(
        "Tracking start frame",
        clip_start_frame,
        clip_end_frame,
        previous_frame_index,
    )
    st.session_state["frame_index"] = frame_index

    frame_rgb = read_frame(str(video_path), frame_index)
    frame_height, frame_width = frame_rgb.shape[:2]
    obj_projected_points: np.ndarray | None = None
    obj_faces: list[list[int]] = []
    obj_track_count = 0
    obj_overlay_enabled = uploaded_obj is not None
    obj_max_points = 80
    if uploaded_obj is not None:
        try:
            obj_vertices, obj_faces = parse_obj_model(uploaded_obj.getvalue())
        except RuntimeError as error:
            st.error(str(error))
            obj_overlay_enabled = False
            obj_vertices = np.empty((0, 3), dtype=np.float32)
        if obj_overlay_enabled:
            with st.expander("3D model placement", expanded=True):
                use_mouse_obj_placement = st.checkbox("Move/zoom 3D model with mouse", value=True)
                if use_mouse_obj_placement:
                    placement_frame, placement_scale = resize_for_canvas(frame_rgb)
                    placement_height, placement_width = placement_frame.shape[:2]
                    placement_result = st_canvas(
                        fill_color="rgba(60, 220, 255, 0.08)",
                        stroke_width=3,
                        stroke_color="#3cdcff",
                        background_image=Image.fromarray(placement_frame),
                        update_streamlit=True,
                        height=placement_height,
                        width=placement_width,
                        drawing_mode="transform",
                        initial_drawing=obj_control_box_drawing(placement_width, placement_height),
                        display_toolbar=False,
                        key=f"obj_placement_{video_path.name}_{frame_index}_{uploaded_obj.name}",
                    )
                    obj_center_x, obj_center_y, obj_scale = obj_placement_from_canvas(
                        placement_result.json_data,
                        placement_scale,
                        frame_width,
                        frame_height,
                    )
                    st.caption("Move the blue box to translate the model. Resize it to zoom.")
                else:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        obj_center_x = st.slider("3D model X", 0, frame_width, frame_width // 2, 1)
                    with col_b:
                        obj_center_y = st.slider("3D model Y", 0, frame_height, frame_height // 2, 1)
                    with col_c:
                        obj_scale = st.slider(
                            "3D model scale px",
                            10,
                            max(20, max(frame_width, frame_height)),
                            max(40, min(frame_width, frame_height) // 4),
                            5,
                        )
                rot_a, rot_b, rot_c = st.columns(3)
                with rot_a:
                    obj_rotate_x = st.slider("Rotate X", -180, 180, 0, 1)
                with rot_b:
                    obj_rotate_y = st.slider("Rotate Y", -180, 180, 0, 1)
                with rot_c:
                    obj_rotate_z = st.slider("Rotate Z", -180, 180, 0, 1)
                obj_max_points = st.slider("3D model tracking points", 5, 500, 80, 5)
                st.caption(
                    "This is a 2D orthographic projection of the OBJ. The generated points are tracked like normal points."
                )
            obj_projected_points = project_obj_vertices(
                obj_vertices,
                frame_width,
                frame_height,
                obj_center_x,
                obj_center_y,
                obj_scale,
                obj_rotate_x,
                obj_rotate_y,
                obj_rotate_z,
            )

    canvas_background = frame_rgb
    if obj_overlay_enabled and obj_projected_points is not None:
        canvas_background = draw_obj_overlay(frame_rgb, obj_projected_points, obj_faces)

    display_frame, canvas_scale = resize_for_canvas(canvas_background)
    height, width = display_frame.shape[:2]

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Select points or regions")
        canvas_result = st_canvas(
            fill_color="rgba(255, 72, 92, 0.20)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=Image.fromarray(display_frame),
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode=mode,
            point_display_radius=6,
            display_toolbar=True,
            key=f"canvas_{video_path.name}_{frame_index}_{mode}",
        )

    tracks, labels = parse_annotations(canvas_result.json_data, canvas_scale)
    manual_point_count = sum(len(track) for track in tracks)
    if obj_overlay_enabled and obj_projected_points is not None:
        obj_tracks, obj_labels = obj_tracks_from_projection(
            obj_projected_points,
            frame_width,
            frame_height,
            obj_max_points,
        )
        obj_track_count = len(obj_tracks)
        tracks.extend(obj_tracks)
        labels.extend(obj_labels)
    grid_point_count = 0
    if add_point_cloud:
        frame_height, frame_width = frame_rgb.shape[:2]
        point_cloud_regions = None
        if point_cloud_area == "Drawn rect/polygon areas":
            point_cloud_regions = parse_annotation_regions(canvas_result.json_data, canvas_scale)
            if not point_cloud_regions:
                st.warning("Draw at least one rectangle or polygon to place the point cloud inside it.")
        if point_cloud_area == "Drawn rect/polygon areas" and not point_cloud_regions:
            grid_tracks, grid_labels = [], []
        else:
            grid_tracks, grid_labels = generate_grid_tracks(
                frame_width,
                frame_height,
                grid_spacing,
                grid_margin,
                grid_max_points,
                point_cloud_regions,
            )
        grid_point_count = len(grid_tracks)
        tracks.extend(grid_tracks)
        labels.extend(grid_labels)

    with right:
        st.subheader("Preview")
        if tracks:
            preview_frame = frame_rgb
            if obj_overlay_enabled and obj_projected_points is not None:
                preview_frame = draw_obj_overlay(preview_frame, obj_projected_points, obj_faces)
            st.image(draw_tracks(preview_frame, tracks, labels), channels="RGB", use_container_width=True)
            st.caption(
                f"{manual_point_count} manual point(s), {obj_track_count} 3D model point(s), "
                f"{grid_point_count} grid point(s), "
                f"{sum(len(track) for track in tracks)} total tracked point(s)"
            )
            total_point_count = sum(len(track) for track in tracks)
            uses_local_neural_tracker = any(tracker in LOCAL_NEURAL_TRACKERS for tracker in selected_trackers)
            if uses_local_neural_tracker and (total_point_count > 100 or clip_duration > 60):
                st.warning(
                    "This can be slow with CoTracker/LiteTracker. For faster runs use a shorter interval, "
                    "Fast point cloud, fewer max points, larger spacing, or OpenCV Lucas-Kanade."
                )
        else:
            preview_frame = frame_rgb
            if obj_overlay_enabled and obj_projected_points is not None:
                preview_frame = draw_obj_overlay(preview_frame, obj_projected_points, obj_faces)
            st.image(preview_frame, channels="RGB", use_container_width=True)
            st.caption("Draw points, rectangles, or polygons on the left.")

        if compare_mode:
            if selected_trackers:
                st.info(f"Comparison will run: {', '.join(selected_trackers)}")
            else:
                st.warning("Select at least two models to compare.")
            for selected_tracker in selected_trackers:
                if selected_tracker in {SAM2_TRACKER, SURGISAM2_TRACKER, SAM3_TRACKER, MEDSAM2_TRACKER}:
                    st.info(external_tracker_setup_instructions(selected_tracker))
            if COTRACKER_TRACKER in selected_trackers:
                st.info("First CoTracker3 run may pause while the PyTorch Hub model loads.")
            if COTRACKER_OFFLINE_TRACKER in selected_trackers:
                st.info("CoTracker3 Offline loads the whole selected clip into memory before tracking.")
            if LITETRACKER_TRACKER in selected_trackers:
                st.info("LiteTracker uses the cloned official repo and the selected `.pth` weights.")
        else:
            if tracker_name == COTRACKER_TRACKER:
                st.info("First CoTracker3 run may pause while the PyTorch Hub model loads.")
            elif tracker_name == COTRACKER_OFFLINE_TRACKER:
                st.info("CoTracker3 Offline loads the whole selected clip into memory before tracking.")
            elif tracker_name == LITETRACKER_TRACKER:
                st.info("LiteTracker uses the cloned official repo and the selected `.pth` weights.")
            elif tracker_name in {SAM2_TRACKER, SURGISAM2_TRACKER, SAM3_TRACKER, MEDSAM2_TRACKER}:
                st.info(external_tracker_setup_instructions(tracker_name))

        can_start = bool(tracks) and (not compare_mode or len(selected_trackers) >= 2)
        if (
            can_start
            and enable_gpu_local_neural
            and any(tracker in LOCAL_NEURAL_TRACKERS for tracker in selected_trackers)
            and sum(len(track) for track in tracks) > 100
        ):
            st.warning("GPU local tracking with more than 100 points is likely to crash. Use fewer points first.")
        if compare_mode and selected_trackers and len(selected_trackers) < 2:
            st.warning("Pick at least two models for a collage comparison.")
        start_tracking = st.button(
            "Compare" if compare_mode else "Track",
            type="primary",
            disabled=not can_start,
        )

    if start_tracking:
        source_start_frame = frame_index
        source_end_frame = clip_end_frame
        tracking_video_path = video_path
        tracking_start_frame = frame_index
        tracking_end_frame = source_end_frame
        should_create_clip = use_clip_interval and (clip_start_frame > 0 or clip_end_frame < last_frame)
        if should_create_clip:
            with st.spinner("Creating selected video interval..."):
                tracking_video_path = trim_video_interval(video_path, clip_start_frame, clip_end_frame, fps)
            transient_input_paths.append(tracking_video_path)
            tracking_start_frame = frame_index - clip_start_frame
            tracking_end_frame = clip_end_frame - clip_start_frame

        end_frame = source_end_frame
        test_mode = "comparison" if compare_mode else "single"
        test_dir = create_test_output_dir(video_path, frame_index, test_mode)
        log_path = timing_log_path(test_dir)
        run_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        annotation_count = len(tracks)
        point_count = sum(len(track) for track in tracks)
        st.divider()
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        status_placeholder.caption(
            f"Running test from source frames {source_start_frame} to {source_end_frame} at {fps:.2f} fps."
        )

        if compare_mode:
            st.subheader("Comparing models")
            completed_outputs: list[tuple[str, Path]] = []
            for index, selected_tracker in enumerate(selected_trackers, start=1):
                result_path = output_video_path(test_dir, selected_tracker)
                run_device_name = tracker_device_for_run(
                    selected_tracker,
                    device_name,
                    enable_gpu_local_neural,
                )
                if run_device_name != device_name:
                    st.info(f"{selected_tracker}: using CPU because GPU for CoTracker/LiteTracker is disabled.")
                status_placeholder.caption(
                    f"[{index}/{len(selected_trackers)}] Running {selected_tracker} on {run_device_name}..."
                )
                start_time = time.perf_counter()
                saved_path: Path | None = None
                status = "failed"
                error_message = ""
                try:
                    saved_path = run_tracker_model(
                        selected_tracker,
                        tracking_video_path,
                        tracking_start_frame,
                        tracking_end_frame,
                        tracks,
                        labels,
                        fps,
                        frame_skip,
                        model_max_side,
                        lite_weights_path,
                        cotracker_offline_chunk_frames,
                        run_device_name,
                        external_commands,
                        result_path,
                        frame_placeholder,
                        status_placeholder,
                        show_live_preview,
                        freeze_lost_points,
                        instrument_avoidance,
                        track_validation,
                    )
                except RuntimeError as error:
                    error_message = format_tracker_error(error)
                    st.error(f"{selected_tracker} failed:\n\n{error_message}")
                duration = time.perf_counter() - start_time

                if saved_path and Path(saved_path).exists():
                    status = "success"
                    completed_outputs.append((selected_tracker, Path(saved_path)))
                    st.success(f"{selected_tracker} finished in {duration:.2f}s")
                else:
                    if not error_message:
                        error_message = "No output video was created."
                        st.error(f"{selected_tracker} finished, but no output video was created.")

                append_timing_log(
                    test_dir,
                    {
                        "timestamp": run_timestamp,
                        "video": str(video_path),
                        "mode": "comparison",
                        "model": selected_tracker,
                        "device": run_device_name,
                        "start_frame": frame_index,
                        "end_frame": end_frame,
                        "annotation_count": annotation_count,
                        "point_count": point_count,
                        "status": status,
                        "duration_seconds": f"{duration:.3f}",
                        "output_path": str(saved_path or result_path),
                        "error": error_message,
                    },
                )

            if len(completed_outputs) < 2:
                status_placeholder.error("Need at least two successful model outputs to build a comparison collage.")
                cleanup_paths([test_dir])
                st.stop()

            collage_path = comparison_video_path(test_dir)
            collage_start_time = time.perf_counter()
            try:
                collage_saved_path = create_comparison_collage(
                    completed_outputs,
                    collage_path,
                    fps,
                    collage_tile_width,
                    show_live_preview,
                    frame_placeholder,
                    status_placeholder,
                )
            except RuntimeError as error:
                append_timing_log(
                    test_dir,
                    {
                        "timestamp": run_timestamp,
                        "video": str(video_path),
                        "mode": "comparison",
                        "model": "Comparison Collage",
                        "device": device_name,
                        "start_frame": frame_index,
                        "end_frame": end_frame,
                        "annotation_count": annotation_count,
                        "point_count": point_count,
                        "status": "failed",
                        "duration_seconds": f"{time.perf_counter() - collage_start_time:.3f}",
                        "output_path": str(collage_path),
                        "error": str(error),
                    },
                )
                cleanup_paths([test_dir])
                st.error(str(error))
                st.stop()
            append_timing_log(
                test_dir,
                {
                    "timestamp": run_timestamp,
                    "video": str(video_path),
                    "mode": "comparison",
                    "model": "Comparison Collage",
                    "device": device_name,
                    "start_frame": frame_index,
                    "end_frame": end_frame,
                    "annotation_count": annotation_count,
                    "point_count": point_count,
                    "status": "success",
                    "duration_seconds": f"{time.perf_counter() - collage_start_time:.3f}",
                    "output_path": str(collage_saved_path),
                    "error": "",
                },
            )
            collage_preview_path = make_streamlit_preview_video(Path(collage_saved_path))
            collage_preview_bytes = Path(collage_preview_path).read_bytes()
            collage_bytes = Path(collage_saved_path).read_bytes()
            timing_bytes = log_path.read_bytes() if log_path.exists() else b""
            status_placeholder.success("Comparison collage finished. Local working files will be removed.")
            st.video(collage_preview_bytes)
            st.download_button(
                "Download comparison collage",
                data=collage_bytes,
                file_name=Path(collage_saved_path).name,
                mime="video/mp4",
            )
            if timing_bytes:
                st.download_button(
                    "Download timing CSV",
                    data=timing_bytes,
                    file_name=log_path.name,
                    mime="text/csv",
                )
            cleanup_paths([test_dir])
        else:
            result_path = output_video_path(test_dir, tracker_name)
            st.subheader(f"Tracking with {tracker_name}")
            run_device_name = tracker_device_for_run(
                tracker_name,
                device_name,
                enable_gpu_local_neural,
            )
            if run_device_name != device_name:
                st.info(f"{tracker_name}: using CPU because GPU for CoTracker/LiteTracker is disabled.")
            start_time = time.perf_counter()
            saved_path: Path | None = None
            try:
                saved_path = run_tracker_model(
                    tracker_name,
                    tracking_video_path,
                    tracking_start_frame,
                    tracking_end_frame,
                    tracks,
                    labels,
                    fps,
                    frame_skip,
                    model_max_side,
                    lite_weights_path,
                    cotracker_offline_chunk_frames,
                    run_device_name,
                    external_commands,
                    result_path,
                    frame_placeholder,
                    status_placeholder,
                    show_live_preview,
                    freeze_lost_points,
                    instrument_avoidance,
                    track_validation,
                )
            except RuntimeError as error:
                error_message = format_tracker_error(error)
                append_timing_log(
                    test_dir,
                    {
                        "timestamp": run_timestamp,
                        "video": str(video_path),
                        "mode": "single",
                        "model": tracker_name,
                        "device": run_device_name,
                        "start_frame": frame_index,
                        "end_frame": end_frame,
                        "annotation_count": annotation_count,
                        "point_count": point_count,
                        "status": "failed",
                        "duration_seconds": f"{time.perf_counter() - start_time:.3f}",
                        "output_path": str(result_path),
                        "error": error_message,
                    },
                )
                cleanup_paths([test_dir])
                st.error(error_message)
                st.stop()

            duration = time.perf_counter() - start_time
            if saved_path and Path(saved_path).exists():
                append_timing_log(
                    test_dir,
                    {
                        "timestamp": run_timestamp,
                        "video": str(video_path),
                        "mode": "single",
                        "model": tracker_name,
                        "device": run_device_name,
                        "start_frame": frame_index,
                        "end_frame": end_frame,
                        "annotation_count": annotation_count,
                        "point_count": point_count,
                        "status": "success",
                        "duration_seconds": f"{duration:.3f}",
                        "output_path": str(saved_path),
                        "error": "",
                    },
                )
                preview_path = make_streamlit_preview_video(Path(saved_path))
                preview_bytes = Path(preview_path).read_bytes()
                result_bytes = Path(saved_path).read_bytes()
                timing_bytes = log_path.read_bytes() if log_path.exists() else b""
                status_placeholder.success(
                    f"Tracking finished in {duration:.2f}s. Local working files will be removed."
                )
                st.video(preview_bytes)
                st.download_button(
                    "Download tracked video",
                    data=result_bytes,
                    file_name=Path(saved_path).name,
                    mime="video/mp4",
                )
                if timing_bytes:
                    st.download_button(
                        "Download timing CSV",
                        data=timing_bytes,
                        file_name=log_path.name,
                        mime="text/csv",
                    )
            else:
                append_timing_log(
                    test_dir,
                    {
                        "timestamp": run_timestamp,
                        "video": str(video_path),
                        "mode": "single",
                        "model": tracker_name,
                        "device": run_device_name,
                        "start_frame": frame_index,
                        "end_frame": end_frame,
                        "annotation_count": annotation_count,
                        "point_count": point_count,
                        "status": "failed",
                        "duration_seconds": f"{duration:.3f}",
                        "output_path": str(result_path),
                        "error": "No output video was created.",
                    },
                )
                status_placeholder.error("Tracking finished, but no output video was created.")
            cleanup_paths([test_dir])
finally:
    cleanup_paths(transient_input_paths)
    cleanup_empty_work_dirs()
