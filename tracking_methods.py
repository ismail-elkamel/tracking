from __future__ import annotations

import json
import shutil
import shlex
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


MODEL_DIR = Path("models")
LITETRACKER_DIR = Path("external/lite-tracker")
LITETRACKER_WEIGHTS_URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth"
TAPNET_DIR = Path("external/tapnet")
TAPIR_WEIGHTS_URL = "https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt"
BOOTSTAPIR_WEIGHTS_URL = "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"

OPENCV_TRACKER = "OpenCV Lucas-Kanade"
COTRACKER_TRACKER = "CoTracker3 Online"
COTRACKER_OFFLINE_TRACKER = "CoTracker3 Offline"
LITETRACKER_TRACKER = "LiteTracker"
TAPIR_TRACKER = "TAPIR"
BOOTSTAPIR_TRACKER = "BootsTAPIR"
SAM2_TRACKER = "SAM2"
SURGISAM2_TRACKER = "SurgiSAM2"
SAM3_TRACKER = "SAM3"
MEDSAM2_TRACKER = "MedSAM2"
CUDA_DEVICE = "CUDA GPU"
CPU_DEVICE = "CPU"


@dataclass(frozen=True)
class InstrumentAvoidanceConfig:
    onnx_path: str
    image_size: int = 512
    threshold: float = 0.4
    dilation: int = 7
    device_name: str = CPU_DEVICE


@dataclass(frozen=True)
class TrackValidationConfig:
    edge_margin: int = 8
    max_jump_px: float = 80.0
    content_margin: int = 24
    black_threshold: int = 12


@dataclass(frozen=True)
class ObjOverlayMetadata:
    anchor_points: np.ndarray
    model_points: np.ndarray
    anchor_points_3d: np.ndarray
    model_points_3d: np.ndarray
    faces: list[list[int]]
    face_colors: list[tuple[int, int, int]]
    edges: list[tuple[int, int]]
    edge_colors: list[tuple[int, int, int]]
    transform_mode: str = "Similarity"
    frame_width: int = 1
    frame_height: int = 1
    pnp_reprojection_error: float = 8.0
    pnp_min_inliers: int = 6
    show_anchor_points: bool = True
    render_style: str = "Wireframe"


OBJ_OVERLAYS: dict[str, ObjOverlayMetadata] = {}
OBJ_PNP_POSE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def obj_edges_from_faces(
    faces: list[list[int]],
    face_colors: list[tuple[int, int, int]] | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    edge_colors_by_key: dict[tuple[int, int], tuple[int, int, int]] = {}
    default_color = (60, 220, 255)
    for face_index, face in enumerate(faces):
        if len(face) < 2:
            continue
        face_color = default_color
        if face_colors is not None and face_index < len(face_colors):
            face_color = face_colors[face_index]
        for start, end in zip(face, face[1:] + face[:1]):
            if start == end:
                continue
            edge = (start, end) if start < end else (end, start)
            edge_colors_by_key.setdefault(edge, face_color)
    edges = sorted(edge_colors_by_key)
    edge_colors = [edge_colors_by_key[edge] for edge in edges]
    return edges, edge_colors


def register_obj_overlay(
    label: str,
    anchor_points: np.ndarray,
    model_points: np.ndarray,
    anchor_points_3d: np.ndarray,
    model_points_3d: np.ndarray,
    faces: list[list[int]],
    face_colors: list[tuple[int, int, int]] | None = None,
    transform_mode: str = "Similarity",
    frame_size: tuple[int, int] = (1, 1),
    pnp_reprojection_error: float = 8.0,
    pnp_min_inliers: int = 6,
    show_anchor_points: bool = True,
    render_style: str = "Wireframe",
) -> None:
    OBJ_PNP_POSE_CACHE.pop(label, None)
    if face_colors is None:
        face_colors = [(60, 220, 255)] * len(faces)
    edges, edge_colors = obj_edges_from_faces(faces, face_colors)
    OBJ_OVERLAYS[label] = ObjOverlayMetadata(
        anchor_points=anchor_points.astype(np.float32),
        model_points=model_points.astype(np.float32),
        anchor_points_3d=anchor_points_3d.astype(np.float32),
        model_points_3d=model_points_3d.astype(np.float32),
        faces=faces,
        face_colors=face_colors,
        edges=edges,
        edge_colors=edge_colors,
        transform_mode=transform_mode,
        frame_width=int(frame_size[0]),
        frame_height=int(frame_size[1]),
        pnp_reprojection_error=float(pnp_reprojection_error),
        pnp_min_inliers=int(pnp_min_inliers),
        show_anchor_points=bool(show_anchor_points),
        render_style=render_style,
    )


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


def default_tapir_weights_path(tracker_name: str) -> Path:
    if tracker_name == TAPIR_TRACKER:
        return MODEL_DIR / "tapir_checkpoint_panning.pt"
    return MODEL_DIR / "bootstapir_checkpoint_v2.pt"


def download_tapir_weights(tracker_name: str, path: Path) -> None:
    url = TAPIR_WEIGHTS_URL if tracker_name == TAPIR_TRACKER else BOOTSTAPIR_WEIGHTS_URL
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.pt")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(path)


def tracker_slug(tracker_name: str) -> str:
    return tracker_name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def default_external_command(tracker_name: str) -> str:
    if tracker_name == TAPIR_TRACKER:
        return (
            "python external/TAPIR/infer_prompts.py "
            "--video {video} --start-frame {start_frame} --end-frame {end_frame} "
            "--prompts {prompts} --output {output} --device {device} "
            "--model-profile tapir --checkpoint models/tapir_checkpoint_panning.pt "
            "--auto-download --repo-path external/tapnet --resize-size 256 "
            "--query-chunk-size 32 {freeze_lost_points}"
        )
    if tracker_name == BOOTSTAPIR_TRACKER:
        return (
            "python external/TAPIR/infer_prompts.py "
            "--video {video} --start-frame {start_frame} --end-frame {end_frame} "
            "--prompts {prompts} --output {output} --device {device} "
            "--model-profile bootstapir --checkpoint models/bootstapir_checkpoint_v2.pt "
            "--auto-download --repo-path external/tapnet --resize-size 256 "
            "--query-chunk-size 32 {freeze_lost_points}"
        )
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


def external_tracker_adapter_path(tracker_name: str) -> Path | None:
    if tracker_name in {TAPIR_TRACKER, BOOTSTAPIR_TRACKER}:
        return Path("external/TAPIR/infer_prompts.py")
    if tracker_name in {SAM2_TRACKER, SURGISAM2_TRACKER}:
        return Path("external/Surgical-SAM-2/infer_prompts.py")
    if tracker_name == SAM3_TRACKER:
        return Path("external/SAM3/infer_prompts.py")
    if tracker_name == MEDSAM2_TRACKER:
        return Path("external/MedSAM2/infer_prompts.py")
    return None


def external_tracker_is_available(tracker_name: str) -> bool:
    adapter_path = external_tracker_adapter_path(tracker_name)
    return adapter_path is None or adapter_path.exists()


def unavailable_external_tracker_message(tracker_name: str) -> str:
    adapter_path = external_tracker_adapter_path(tracker_name)
    if adapter_path is None:
        return ""
    return (
        f"{tracker_name} is hidden because `{adapter_path}` is missing. "
        "The public external repo is present, but this app also needs its local "
        "`infer_prompts.py` adapter."
    )


def external_tracker_setup_instructions(tracker_name: str) -> str:
    if tracker_name in {TAPIR_TRACKER, BOOTSTAPIR_TRACKER}:
        checkpoint = default_tapir_weights_path(tracker_name)
        return (
            f"{tracker_name} uses Google DeepMind TAPNet's PyTorch TAPIR implementation. "
            "Install it with `git clone https://github.com/google-deepmind/tapnet external/tapnet` "
            "then `python -m pip install -e external/tapnet[torch]`. "
            f"Download the checkpoint to `{checkpoint}` with the sidebar button."
        )
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


def estimate_obj_transform(label: str, tracked_points: np.ndarray) -> np.ndarray | None:
    metadata = OBJ_OVERLAYS.get(label)
    if metadata is None:
        return None
    source = metadata.anchor_points
    target = tracked_points.astype(np.float32)
    if len(source) < 1 or len(target) < 1:
        return None
    count = min(len(source), len(target))
    if count == 1:
        dx, dy = (target[0] - source[0]).astype(float)
        return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    if count == 2:
        source_delta = source[1] - source[0]
        target_delta = target[1] - target[0]
        source_length = float(np.linalg.norm(source_delta))
        target_length = float(np.linalg.norm(target_delta))
        if source_length <= 1e-6 or target_length <= 1e-6:
            dx, dy = (target[0] - source[0]).astype(float)
            return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        scale = target_length / source_length
        source_angle = float(np.arctan2(source_delta[1], source_delta[0]))
        target_angle = float(np.arctan2(target_delta[1], target_delta[0]))
        angle = target_angle - source_angle
        cos_a = float(np.cos(angle)) * scale
        sin_a = float(np.sin(angle)) * scale
        transform = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]], dtype=np.float32)
        transform[:, 2] = target[0] - (transform[:, :2] @ source[0])
        return transform
    transform, _ = cv2.estimateAffinePartial2D(
        source[:count],
        target[:count],
        method=cv2.RANSAC,
        ransacReprojThreshold=8.0,
    )
    if transform is None:
        transform = cv2.getAffineTransform(
            np.float32([source[0], source[count // 2], source[count - 1]]),
            np.float32([target[0], target[count // 2], target[count - 1]]),
        ) if count >= 3 else None
    return transform


def apply_obj_transform(points: np.ndarray, transform: np.ndarray | None) -> np.ndarray:
    if transform is None:
        return points.astype(np.float32)
    ones = np.ones((len(points), 1), dtype=np.float32)
    homogeneous = np.hstack([points.astype(np.float32), ones])
    return (homogeneous @ transform.T).astype(np.float32)


def obj_camera_matrix(metadata: ObjOverlayMetadata) -> np.ndarray:
    focal = float(max(metadata.frame_width, metadata.frame_height))
    return np.array(
        [
            [focal, 0.0, metadata.frame_width / 2.0],
            [0.0, focal, metadata.frame_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def estimate_obj_pnp_pose(
    metadata: ObjOverlayMetadata,
    tracked_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    count = min(len(metadata.anchor_points_3d), len(tracked_points))
    if count < 6:
        return None

    object_points = metadata.anchor_points_3d[:count].astype(np.float32)
    image_points = tracked_points[:count].astype(np.float32)
    valid = np.isfinite(image_points).all(axis=1)
    object_points = object_points[valid]
    image_points = image_points[valid]
    required_inliers = max(6, int(metadata.pnp_min_inliers))
    if len(object_points) < required_inliers:
        return None

    camera_matrix = obj_camera_matrix(metadata)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        iterationsCount=200,
        reprojectionError=float(metadata.pnp_reprojection_error),
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or inliers is None:
        return None
    inliers = inliers.reshape(-1)
    if len(inliers) < required_inliers:
        return None
    return rvec, tvec, inliers


def project_obj_points_with_pnp(
    metadata: ObjOverlayMetadata,
    tracked_points: np.ndarray,
    object_points: np.ndarray,
) -> np.ndarray | None:
    pose = estimate_obj_pnp_pose(metadata, tracked_points)
    if pose is None:
        return None
    return project_obj_points_from_pose(metadata, object_points, pose)


def project_obj_points_from_pose(
    metadata: ObjOverlayMetadata,
    object_points: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    rvec, tvec, _ = pose
    projected, _ = cv2.projectPoints(
        object_points.astype(np.float32),
        rvec,
        tvec,
        obj_camera_matrix(metadata),
        np.zeros((4, 1), dtype=np.float32),
    )
    return projected.reshape(-1, 2).astype(np.float32)


def estimate_cached_obj_pnp_pose(
    label: str,
    metadata: ObjOverlayMetadata,
    tracked_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    pose = estimate_obj_pnp_pose(metadata, tracked_points)
    if pose is not None:
        rvec, tvec, inliers = pose
        OBJ_PNP_POSE_CACHE[label] = (rvec.copy(), tvec.copy())
        return rvec, tvec, inliers

    cached_pose = OBJ_PNP_POSE_CACHE.get(label)
    if cached_pose is None:
        return None
    rvec, tvec = cached_pose
    return rvec, tvec, np.empty(0, dtype=np.int32)


def project_obj_with_pnp(label: str, tracked_points: np.ndarray) -> np.ndarray | None:
    metadata = OBJ_OVERLAYS.get(label)
    if metadata is None:
        return None
    return project_obj_points_with_pnp(metadata, tracked_points, metadata.model_points_3d)


def project_obj_anchors_with_pnp(label: str, tracked_points: np.ndarray) -> np.ndarray | None:
    metadata = OBJ_OVERLAYS.get(label)
    if metadata is None:
        return None
    return project_obj_points_with_pnp(metadata, tracked_points, metadata.anchor_points_3d)


def transform_obj_model_points(label: str, tracked_points: np.ndarray) -> np.ndarray:
    metadata = OBJ_OVERLAYS.get(label)
    if metadata is None:
        return tracked_points
    if metadata.transform_mode == "PnP":
        projected = project_obj_with_pnp(label, tracked_points)
        if projected is not None:
            return projected
    transform = estimate_obj_transform(label, tracked_points)
    return apply_obj_transform(metadata.model_points, transform)


def apply_instrument_occlusion(
    output: np.ndarray,
    base_frame: np.ndarray,
    instrument_mask: np.ndarray | None,
) -> np.ndarray:
    if instrument_mask is None:
        return output
    mask = instrument_mask.astype(bool, copy=False)
    if mask.shape[:2] != output.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (output.shape[1], output.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    output[mask] = base_frame[mask]
    return output


def draw_obj_mesh(
    output: np.ndarray,
    label: str,
    tracked_points: np.ndarray,
    instrument_mask: np.ndarray | None = None,
) -> np.ndarray:
    base_frame = output.copy()
    metadata = OBJ_OVERLAYS.get(label)
    if metadata is not None and metadata.transform_mode == "PnP":
        pose = estimate_cached_obj_pnp_pose(label, metadata, tracked_points)
        if pose is not None:
            points = project_obj_points_from_pose(metadata, metadata.model_points_3d, pose)
            anchor_points = project_obj_points_from_pose(metadata, metadata.anchor_points_3d, pose)
        else:
            transform = estimate_obj_transform(label, tracked_points)
            points = apply_obj_transform(metadata.model_points, transform)
            anchor_points = apply_obj_transform(metadata.anchor_points, transform)
    else:
        transform = estimate_obj_transform(label, tracked_points)
        points = apply_obj_transform(metadata.model_points, transform) if metadata is not None else tracked_points
        anchor_points = apply_obj_transform(metadata.anchor_points, transform) if metadata is not None else tracked_points
    faces = metadata.faces if metadata is not None else []
    edges = metadata.edges if metadata is not None else []
    face_colors = metadata.face_colors if metadata is not None else []
    edge_colors = metadata.edge_colors if metadata is not None else []
    if len(points) < 3:
        return output

    height, width = output.shape[:2]
    render_style = metadata.render_style if metadata is not None else "Wireframe"
    if render_style == "Wireframe":
        rendered_edges = 0
        for edge_index, (start, end) in enumerate(edges):
            if start < 0 or end < 0 or start >= len(points) or end >= len(points):
                continue
            p1 = np.round(points[start]).astype(np.int32)
            p2 = np.round(points[end]).astype(np.int32)
            p1_in_frame = 0 <= p1[0] < width and 0 <= p1[1] < height
            p2_in_frame = 0 <= p2[0] < width and 0 <= p2[1] < height
            if not (p1_in_frame or p2_in_frame):
                continue
            edge_color = edge_colors[edge_index] if edge_index < len(edge_colors) else (60, 220, 255)
            cv2.line(output, tuple(p1), tuple(p2), edge_color, 1, lineType=cv2.LINE_AA)
            rendered_edges += 1
        if rendered_edges == 0:
            pts = np.round(points).astype(np.int32)
            in_frame = (
                (pts[:, 0] >= 0)
                & (pts[:, 0] < width)
                & (pts[:, 1] >= 0)
                & (pts[:, 1] < height)
            )
            for point in pts[in_frame][:: max(1, len(pts) // 1000)]:
                cv2.circle(output, tuple(point), 1, (60, 220, 255), -1, lineType=cv2.LINE_AA)
        if metadata is None or metadata.show_anchor_points:
            for point in np.round(anchor_points[:: max(1, len(anchor_points) // 260)]).astype(np.int32):
                if 0 <= point[0] < width and 0 <= point[1] < height:
                    cv2.circle(output, tuple(point), 6, (255, 72, 92), 2, lineType=cv2.LINE_AA)
        return apply_instrument_occlusion(output, base_frame, instrument_mask)

    overlay = output.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    rendered_faces = 0
    for face_index, face in enumerate(faces):
        valid_face = [index for index in face if 0 <= index < len(points)]
        if len(valid_face) < 3:
            continue
        polygon = np.round(points[valid_face]).astype(np.int32)
        in_frame = (
            (polygon[:, 0] >= 0)
            & (polygon[:, 0] < width)
            & (polygon[:, 1] >= 0)
            & (polygon[:, 1] < height)
        )
        if not bool(in_frame.any()):
            continue
        face_color = face_colors[face_index] if face_index < len(face_colors) else (60, 220, 255)
        cv2.fillPoly(overlay, [polygon], face_color, lineType=cv2.LINE_AA)
        cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_AA)
        cv2.polylines(overlay, [polygon], True, face_color, 1, lineType=cv2.LINE_AA)
        rendered_faces += 1

    if rendered_faces == 0:
        pts = np.round(points).astype(np.int32)
        in_frame = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] < width)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < height)
        )
        pts = pts[in_frame]
        if len(pts) < 3:
            return output
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(overlay, hull, (60, 220, 255), lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)
    output[mask > 0] = blended[mask > 0]
    if metadata is None or metadata.show_anchor_points:
        for point in np.round(anchor_points[:: max(1, len(anchor_points) // 220)]).astype(np.int32):
            if 0 <= point[0] < width and 0 <= point[1] < height:
                cv2.circle(output, tuple(point), 6, (255, 72, 92), 2, lineType=cv2.LINE_AA)
    return apply_instrument_occlusion(output, base_frame, instrument_mask)


def draw_tracks(
    frame_rgb: np.ndarray,
    tracks: list[np.ndarray],
    labels: list[str],
    instrument_mask: np.ndarray | None = None,
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

    for index, points in enumerate(tracks):
        label = labels[index] if index < len(labels) else f"annotation {index + 1}"
        if label.startswith("obj "):
            output = draw_obj_mesh(output, label, points, instrument_mask)

    for index, points in enumerate(tracks):
        color = colors[index % len(colors)]
        label = labels[index] if index < len(labels) else f"annotation {index + 1}"
        if label.startswith("obj "):
            continue
        pts = np.round(points).astype(int)
        if len(pts) == 1:
            cv2.circle(output, tuple(pts[0]), 7, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(output, tuple(pts[0]), 11, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        else:
            closed = len(pts) >= 3
            cv2.polylines(output, [pts], closed, color, 3, lineType=cv2.LINE_AA)
            for point in pts:
                cv2.circle(output, tuple(point), 5, color, -1, lineType=cv2.LINE_AA)
        if len(pts) and label and not label.startswith(("grid ", "obj ")):
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


def onnxruntime_available_providers() -> list[str]:
    preload_onnxruntime_cuda_libraries()
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is required to avoid instruments with the ONNX model. "
            "Install it with `python -m pip install onnxruntime` for CPU or "
            "`python -m pip install onnxruntime-gpu` for CUDA."
        ) from error
    return list(ort.get_available_providers())


def preload_onnxruntime_cuda_libraries() -> None:
    import ctypes

    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia_dir = Path(sys.prefix) / "lib" / python_dir / "site-packages" / "nvidia"
    library_prefixes = [
        "libcudart.so",
        "libnvJitLink.so",
        "libnvrtc.so",
        "libcublas.so",
        "libcublasLt.so",
        "libcufft.so",
        "libcurand.so",
        "libcusparse.so",
        "libcusolver.so",
        "libcudnn.so",
    ]
    loaded_paths: set[Path] = set()
    for library_prefix in library_prefixes:
        for library_path in sorted(nvidia_dir.glob(f"*/lib/{library_prefix}*")):
            if library_path in loaded_paths or not library_path.is_file():
                continue
            try:
                ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
                loaded_paths.add(library_path)
            except OSError:
                continue


def instrument_onnx_providers(device_name: str) -> list[str]:
    available = onnxruntime_available_providers()
    if device_name == CUDA_DEVICE:
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "Instrument ONNX GPU was selected, but ONNXRuntime does not expose "
                "`CUDAExecutionProvider` in this environment. Current providers: "
                f"{available}. Install an ONNXRuntime GPU build in `track_env`, then restart Streamlit."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


@st.cache_resource(show_spinner=False)
def load_instrument_avoidance_session(onnx_path: str, device_name: str):
    preload_onnxruntime_cuda_libraries()
    import onnxruntime as ort

    model_path = Path(onnx_path).expanduser().resolve()
    if not model_path.exists():
        raise RuntimeError(f"Instrument avoidance ONNX model not found: {model_path}")
    session = ort.InferenceSession(
        str(model_path),
        providers=instrument_onnx_providers(device_name),
    )
    if device_name == CUDA_DEVICE and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "Instrument ONNX was requested on GPU, but ONNXRuntime created a CPU session. "
            f"Session providers: {session.get_providers()}."
        )
    return session, session.get_inputs()[0].name


def predict_instrument_mask(
    frame_rgb: np.ndarray,
    config: InstrumentAvoidanceConfig | None,
) -> np.ndarray | None:
    if config is None:
        return None

    session, input_name = load_instrument_avoidance_session(config.onnx_path, config.device_name)
    model_size = int(config.image_size)
    resized = cv2.resize(frame_rgb, (model_size, model_size), interpolation=cv2.INTER_LINEAR)
    array = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    input_array = np.transpose(array, (2, 0, 1))[None].astype(np.float32)
    logits = session.run(None, {input_name: input_array})[0]
    probability = 1.0 / (1.0 + np.exp(-np.asarray(logits)[0, 0]))

    height, width = frame_rgb.shape[:2]
    probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = probability >= float(config.threshold)
    dilation = max(0, int(config.dilation))
    if dilation > 0:
        kernel_size = dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    return mask


def points_outside_instrument(points: np.ndarray, instrument_mask: np.ndarray | None) -> np.ndarray:
    if instrument_mask is None or len(points) == 0:
        return np.ones(len(points), dtype=bool)

    height, width = instrument_mask.shape[:2]
    rounded = np.round(points).astype(int)
    x = np.clip(rounded[:, 0], 0, max(width - 1, 0))
    y = np.clip(rounded[:, 1], 0, max(height - 1, 0))
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    return in_bounds & ~instrument_mask[y, x]


def validate_tracked_points(
    points: np.ndarray,
    last_valid_points: np.ndarray,
    frame_rgb: np.ndarray,
    config: TrackValidationConfig | None,
    base_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if base_valid_mask is None:
        valid = np.ones(len(points), dtype=bool)
    else:
        valid = base_valid_mask.astype(bool, copy=True)
    if config is None or len(points) == 0:
        return valid

    height, width = frame_rgb.shape[:2]
    finite = np.isfinite(points).all(axis=1)
    valid &= finite

    edge_margin = max(0, int(config.edge_margin))
    if edge_margin > 0:
        valid &= (
            (points[:, 0] >= edge_margin)
            & (points[:, 0] < float(width - edge_margin))
            & (points[:, 1] >= edge_margin)
            & (points[:, 1] < float(height - edge_margin))
        )

    content_margin = max(0, int(config.content_margin))
    if content_margin > 0:
        content_pixels = frame_rgb.max(axis=2) > int(config.black_threshold)
        coords = np.argwhere(content_pixels)
        if len(coords):
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            valid &= (
                (points[:, 0] >= float(x0 + content_margin))
                & (points[:, 0] <= float(x1 - content_margin))
                & (points[:, 1] >= float(y0 + content_margin))
                & (points[:, 1] <= float(y1 - content_margin))
            )

    max_jump = float(config.max_jump_px)
    if max_jump > 0 and len(last_valid_points) == len(points):
        jump = np.linalg.norm(points - last_valid_points, axis=1)
        valid &= jump <= max_jump

    return valid


def regroup_visible_points(
    points: np.ndarray,
    group_sizes: list[int],
    visible_points: np.ndarray,
    labels: list[str] | None = None,
) -> list[np.ndarray]:
    grouped: list[np.ndarray] = []
    offset = 0
    for index, size in enumerate(group_sizes):
        group = points[offset : offset + size]
        group_visible = visible_points[offset : offset + size]
        label = labels[index] if labels and index < len(labels) else ""
        if label.startswith("obj "):
            grouped.append(group.astype(np.float32))
        else:
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
    instrument_mask: np.ndarray | None = None,
) -> None:
    drawn = draw_tracks(frame_rgb, tracks, labels, instrument_mask)
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
    instrument_avoidance: InstrumentAvoidanceConfig | None = None,
    track_validation: TrackValidationConfig | None = None,
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
            instrument_mask = None
            if freeze_lost or instrument_avoidance is not None or track_validation is not None:
                instrument_mask = predict_instrument_mask(original_rgb, instrument_avoidance)
                valid_mask = points_outside_instrument(points, instrument_mask)
                valid_mask = validate_tracked_points(
                    points,
                    last_valid_points,
                    original_rgb,
                    track_validation,
                    valid_mask,
                )
                points, last_valid_points, visible_points = filter_visible_points(
                    points,
                    last_valid_points,
                    visible_points,
                    original_rgb,
                    valid_mask,
                )
                grouped_tracks = regroup_visible_points(points, group_sizes, visible_points, labels)
            else:
                grouped_tracks = regroup_points(points, group_sizes)
            emit_tracked_frame(
                original_rgb,
                grouped_tracks,
                labels,
                output_writer,
                show_live_preview,
                frame_placeholder,
                instrument_mask,
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
                if instrument_avoidance is not None or track_validation is not None:
                    instrument_mask = predict_instrument_mask(frame_rgb, instrument_avoidance)
                    valid_mask = points_outside_instrument(flat_points, instrument_mask)
                    valid_mask = validate_tracked_points(
                        flat_points,
                        last_valid_points,
                        frame_rgb,
                        track_validation,
                        valid_mask,
                    )
                    initial_points, last_valid_points, visible_points = filter_visible_points(
                        flat_points,
                        last_valid_points,
                        visible_points,
                        frame_rgb,
                        valid_mask,
                    )
                    initial_tracks = regroup_visible_points(initial_points, group_sizes, visible_points, labels)
                else:
                    initial_tracks = tracks
                emit_tracked_frame(
                    frame_rgb,
                    initial_tracks,
                    labels,
                    output_writer,
                    show_live_preview,
                    frame_placeholder,
                    instrument_mask if instrument_avoidance is not None else None,
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
    instrument_avoidance: InstrumentAvoidanceConfig | None = None,
    track_validation: TrackValidationConfig | None = None,
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
                instrument_mask = None
                if freeze_lost or instrument_avoidance is not None or track_validation is not None:
                    valid_mask = visibility[relative_frame] if visibility is not None else np.ones(len(points), dtype=bool)
                    if instrument_avoidance is not None:
                        instrument_mask = predict_instrument_mask(frame_rgb, instrument_avoidance)
                        valid_mask = valid_mask & points_outside_instrument(points, instrument_mask)
                    valid_mask = validate_tracked_points(
                        points,
                        last_valid_points,
                        frame_rgb,
                        track_validation,
                        valid_mask,
                    )
                    points, last_valid_points, visible_points = filter_visible_points(
                        points,
                        last_valid_points,
                        visible_points,
                        frame_rgb,
                        valid_mask,
                    )
                    grouped_tracks = regroup_visible_points(points, group_sizes, visible_points, labels)
                else:
                    grouped_tracks = regroup_points(points, group_sizes)
                emit_tracked_frame(
                    frame_rgb,
                    grouped_tracks,
                    labels,
                    output_writer,
                    show_live_preview,
                    frame_placeholder,
                    instrument_mask,
                )
                rendered_frame = chunk_start + relative_frame
                status_placeholder.caption(
                    f"CoTracker3 Offline frame {rendered_frame} / {end_frame} on {device}"
                )
                if show_live_preview:
                    sync_to_video_clock(start_time, rendered_frame - start_frame, fps)
                frames_written += 1

            if freeze_lost or instrument_avoidance is not None or track_validation is not None:
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
    instrument_avoidance: InstrumentAvoidanceConfig | None = None,
    track_validation: TrackValidationConfig | None = None,
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
                instrument_mask = None
                if freeze_lost or instrument_avoidance is not None or track_validation is not None:
                    instrument_mask = predict_instrument_mask(frame_rgb, instrument_avoidance)
                    valid_mask = points_outside_instrument(points, instrument_mask)
                    valid_mask = validate_tracked_points(
                        points,
                        last_valid_points,
                        frame_rgb,
                        track_validation,
                        valid_mask,
                    )
                    points, last_valid_points, visible_points = filter_visible_points(
                        points,
                        last_valid_points,
                        visible_points,
                        frame_rgb,
                        valid_mask,
                    )
                    grouped_tracks = regroup_visible_points(points, group_sizes, visible_points, labels)
                else:
                    grouped_tracks = regroup_points(points, group_sizes)
                emit_tracked_frame(
                    frame_rgb,
                    grouped_tracks,
                    labels,
                    output_writer,
                    show_live_preview,
                    frame_placeholder,
                    instrument_mask,
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
    instrument_avoidance: InstrumentAvoidanceConfig | None = None,
    track_validation: TrackValidationConfig | None = None,
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
    instrument_mask = None
    if instrument_avoidance is not None or track_validation is not None:
        instrument_mask = predict_instrument_mask(rgb, instrument_avoidance)
        for index, points in enumerate(active_tracks):
            valid_mask = points_outside_instrument(points, instrument_mask)
            visible_tracks[index] = validate_tracked_points(
                points,
                points,
                rgb,
                track_validation,
                valid_mask,
            )
    displayed_tracks = [
        points[visible_tracks[index]]
        for index, points in enumerate(active_tracks)
    ] if freeze_lost or instrument_avoidance is not None or track_validation is not None else active_tracks
    emit_tracked_frame(
        rgb,
        displayed_tracks,
        labels,
        output_writer,
        show_live_preview,
        frame_placeholder,
        instrument_mask,
    )

    try:
        while current_frame < end_frame:
            target_frame = min(current_frame + frame_skip, end_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ok, next_bgr = cap.read()
            if not ok:
                break

            next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
            next_rgb = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2RGB)
            instrument_mask = predict_instrument_mask(next_rgb, instrument_avoidance)
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
                    if freeze_lost or instrument_avoidance is not None or track_validation is not None:
                        valid_mask = status
                        if instrument_avoidance is not None:
                            valid_mask = valid_mask & points_outside_instrument(candidate_points, instrument_mask)
                        valid_mask = validate_tracked_points(
                            candidate_points,
                            updated,
                            next_rgb,
                            track_validation,
                            valid_mask,
                        )
                        updated, _, visible_tracks[index] = filter_visible_points(
                            candidate_points,
                            updated,
                            visible_tracks[index],
                            next_rgb,
                            valid_mask,
                        )
                    else:
                        updated[status] = candidate_points[status]
                    active_tracks[index] = updated
                elif freeze_lost or instrument_avoidance is not None or track_validation is not None:
                    visible_tracks[index][:] = False

            current_frame = target_frame
            previous_gray = next_gray
            rgb = next_rgb
            displayed_tracks = [
                points[visible_tracks[index]]
                for index, points in enumerate(active_tracks)
            ] if freeze_lost or instrument_avoidance is not None or track_validation is not None else active_tracks
            emit_tracked_frame(
                rgb,
                displayed_tracks,
                labels,
                output_writer,
                show_live_preview,
                frame_placeholder,
                instrument_mask,
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
    end_frame: int,
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
        end_frame=int(end_frame),
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
