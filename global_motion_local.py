from __future__ import annotations

import hashlib
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, send_from_directory, url_for
from werkzeug.utils import secure_filename


TEMP_DIR = Path(tempfile.gettempdir()) / "global_motion_local"
RUN_DIR = TEMP_DIR / "runs"
DEFAULT_OBJ_ROTATION = (-40, 20, -15)
DEFAULT_OBJ_SCALE_FRACTION = 0.5


@dataclass(frozen=True)
class GlobalMotionConfig:
    max_features: int = 2000
    min_inliers: int = 30
    ransac_reprojection_px: float = 5.0
    smoothing: float = 0.25
    max_translation_px: float = 80.0
    max_scale_change: float = 0.12
    max_rotation_deg: float = 8.0


@dataclass(frozen=True)
class ObjMeshPart:
    name: str
    vertices: np.ndarray
    faces: list[list[int]]
    vertex_colors: np.ndarray
    face_colors: list[tuple[int, int, int]]


class StatusCollector:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def caption(self, message: str) -> None:
        self.messages.append(message)


def clamp_rgb(values: list[float]) -> tuple[int, int, int]:
    if max(values) <= 1.0:
        values = [value * 255.0 for value in values]
    return tuple(int(np.clip(round(value), 0, 255)) for value in values[:3])


def default_obj_color(name: str, index: int = 0) -> tuple[int, int, int]:
    lowered = name.lower()
    if any(token in lowered for token in ["artere", "artère", "artery", "aorte", "aorta"]):
        return (255, 72, 92)
    if any(token in lowered for token in ["veine", "vein"]):
        return (72, 160, 255)
    if any(token in lowered for token in ["tumor", "tumeur"]):
        return (255, 96, 210)
    if "cortex" in lowered:
        return (255, 188, 76)
    if any(token in lowered for token in ["rein", "kidney"]):
        return (60, 220, 255)
    palette = [
        (60, 220, 255),
        (255, 188, 76),
        (255, 96, 210),
        (72, 160, 255),
        (118, 214, 90),
        (210, 120, 255),
    ]
    return palette[index % len(palette)]


def material_color(material_name: str, fallback_color: tuple[int, int, int]) -> tuple[int, int, int]:
    if not material_name:
        return fallback_color
    digest = hashlib.sha1(material_name.encode("utf-8", errors="ignore")).digest()
    return (
        int(80 + digest[0] % 176),
        int(80 + digest[1] % 176),
        int(80 + digest[2] % 176),
    )


def parse_mtl_colors(mtl_bytes: bytes) -> dict[str, tuple[int, int, int]]:
    colors: dict[str, tuple[int, int, int]] = {}
    current_material = ""
    text = mtl_bytes.decode("utf-8", errors="ignore")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "newmtl" and len(parts) >= 2:
            current_material = parts[1]
        elif parts[0] == "Kd" and current_material and len(parts) >= 4:
            try:
                colors[current_material] = clamp_rgb([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    return colors


def parse_obj_model(
    obj_bytes: bytes,
    name: str = "model.obj",
    fallback_color: tuple[int, int, int] = (60, 220, 255),
    material_colors: dict[str, tuple[int, int, int]] | None = None,
) -> ObjMeshPart:
    vertices: list[list[float]] = []
    vertex_colors: list[tuple[int, int, int] | None] = []
    faces: list[list[int]] = []
    face_material_colors: list[tuple[int, int, int]] = []
    current_color = fallback_color
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
            color: tuple[int, int, int] | None = None
            if len(parts) >= 7:
                try:
                    color = clamp_rgb([float(parts[4]), float(parts[5]), float(parts[6])])
                except ValueError:
                    color = None
            vertex_colors.append(color)
        elif parts[0] == "usemtl" and len(parts) >= 2:
            current_color = (material_colors or {}).get(parts[1], material_color(parts[1], fallback_color))
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
                index = len(vertices) + index if index < 0 else index - 1
                if index >= 0:
                    face.append(index)
            if len(face) >= 3:
                faces.append(face)
                face_material_colors.append(current_color)
    if not vertices:
        raise RuntimeError(f"{name} does not contain vertices.")

    resolved_vertex_colors = np.asarray(
        [color if color is not None else fallback_color for color in vertex_colors],
        dtype=np.uint8,
    )
    face_colors: list[tuple[int, int, int]] = []
    has_vertex_colors = any(color is not None for color in vertex_colors)
    for face, material_rgb in zip(faces, face_material_colors):
        if has_vertex_colors:
            valid_indices = [index for index in face if 0 <= index < len(resolved_vertex_colors)]
            if valid_indices:
                average = resolved_vertex_colors[valid_indices].astype(np.float32).mean(axis=0)
                face_colors.append(tuple(int(value) for value in np.clip(np.round(average), 0, 255)))
                continue
        face_colors.append(material_rgb)

    return ObjMeshPart(
        name=name,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=faces,
        vertex_colors=resolved_vertex_colors,
        face_colors=face_colors,
    )


def combine_obj_meshes(meshes: list[ObjMeshPart]) -> ObjMeshPart:
    if not meshes:
        raise RuntimeError("No OBJ mesh loaded.")

    vertices: list[np.ndarray] = []
    vertex_colors: list[np.ndarray] = []
    faces: list[list[int]] = []
    face_colors: list[tuple[int, int, int]] = []
    offset = 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        vertex_colors.append(mesh.vertex_colors)
        faces.extend([[index + offset for index in face] for face in mesh.faces])
        face_colors.extend(mesh.face_colors)
        offset += len(mesh.vertices)

    return ObjMeshPart(
        name=" + ".join(mesh.name for mesh in meshes),
        vertices=np.concatenate(vertices, axis=0).astype(np.float32),
        faces=faces,
        vertex_colors=np.concatenate(vertex_colors, axis=0).astype(np.uint8),
        face_colors=face_colors,
    )


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


def project_normalized_obj_vertices(
    normalized_vertices: np.ndarray,
    frame_width: int,
    frame_height: int,
    center_x: float,
    center_y: float,
    scale_px: float,
) -> np.ndarray:
    projected = np.empty((len(normalized_vertices), 2), dtype=np.float32)
    projected[:, 0] = float(center_x) + normalized_vertices[:, 0] * float(scale_px)
    projected[:, 1] = float(center_y) - normalized_vertices[:, 1] * float(scale_px)
    projected[:, 0] = np.clip(projected[:, 0], -frame_width, frame_width * 2)
    projected[:, 1] = np.clip(projected[:, 1], -frame_height, frame_height * 2)
    return projected


def draw_obj_overlay(
    frame_rgb: np.ndarray,
    points_xy: np.ndarray,
    faces: list[list[int]],
    face_colors: list[tuple[int, int, int]],
    render_style: str,
    opacity: float,
) -> np.ndarray:
    output = frame_rgb.copy()
    height, width = output.shape[:2]
    if len(points_xy) == 0:
        return output

    if render_style != "Wireframe":
        overlay = output.copy()
        mask = np.zeros((height, width), dtype=np.uint8)
        for face_index, face in enumerate(faces):
            valid_face = [index for index in face if 0 <= index < len(points_xy)]
            if len(valid_face) < 3:
                continue
            polygon = np.round(points_xy[valid_face]).astype(np.int32)
            in_frame = (
                (polygon[:, 0] >= 0)
                & (polygon[:, 0] < width)
                & (polygon[:, 1] >= 0)
                & (polygon[:, 1] < height)
            )
            if not bool(in_frame.any()):
                continue
            color = face_colors[face_index] if face_index < len(face_colors) else (60, 220, 255)
            cv2.fillPoly(overlay, [polygon], color, lineType=cv2.LINE_AA)
            cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_AA)
            cv2.polylines(overlay, [polygon], True, color, 1, lineType=cv2.LINE_AA)
        alpha = float(np.clip(opacity, 0.0, 1.0))
        blended = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)
        output[mask > 0] = blended[mask > 0]
        return output

    rendered = 0
    for face_index, face in enumerate(faces):
        valid_face = [index for index in face if 0 <= index < len(points_xy)]
        if len(valid_face) < 2:
            continue
        polygon = np.round(points_xy[valid_face]).astype(np.int32)
        in_frame = (
            (polygon[:, 0] >= 0)
            & (polygon[:, 0] < width)
            & (polygon[:, 1] >= 0)
            & (polygon[:, 1] < height)
        )
        if not bool(in_frame.any()):
            continue
        color = face_colors[face_index] if face_index < len(face_colors) else (60, 220, 255)
        cv2.polylines(output, [polygon], True, color, 1, lineType=cv2.LINE_AA)
        rendered += 1

    if rendered == 0:
        pts = np.round(points_xy).astype(np.int32)
        in_frame = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] < width)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < height)
        )
        for point in pts[in_frame][:: max(1, len(pts) // 1000)]:
            cv2.circle(output, tuple(point), 1, (60, 220, 255), -1, lineType=cv2.LINE_AA)
    return output


def affine_to_homogeneous(transform: np.ndarray) -> np.ndarray:
    homogeneous = np.eye(3, dtype=np.float32)
    homogeneous[:2, :] = transform.astype(np.float32)
    return homogeneous


def homogeneous_to_affine(transform: np.ndarray) -> np.ndarray:
    return transform[:2, :].astype(np.float32)


def apply_2d_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1), dtype=np.float32)
    homogeneous = np.hstack([points.astype(np.float32), ones])
    return (homogeneous @ transform.T).astype(np.float32)


def transform_scale_angle(transform: np.ndarray) -> tuple[float, float]:
    a, b = float(transform[0, 0]), float(transform[1, 0])
    scale = float(np.sqrt(a * a + b * b))
    angle = float(np.degrees(np.arctan2(b, a)))
    return scale, angle


def match_global_motion_features(
    previous_gray: np.ndarray,
    next_gray: np.ndarray,
    config: GlobalMotionConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    orb = cv2.ORB_create(
        nfeatures=max(200, int(config.max_features)),
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=12,
    )
    previous_keypoints, previous_desc = orb.detectAndCompute(previous_gray, None)
    next_keypoints, next_desc = orb.detectAndCompute(next_gray, None)
    if previous_desc is None or next_desc is None or len(previous_keypoints) < 4 or len(next_keypoints) < 4:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(previous_desc, next_desc, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good_matches.append(first)

    if len(good_matches) < 4:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, len(good_matches)

    source = np.float32([previous_keypoints[match.queryIdx].pt for match in good_matches])
    target = np.float32([next_keypoints[match.trainIdx].pt for match in good_matches])
    return source, target, len(good_matches)


def estimate_global_motion_transform(
    previous_gray: np.ndarray,
    next_gray: np.ndarray,
    config: GlobalMotionConfig,
) -> tuple[np.ndarray | None, int, int]:
    source, target, match_count = match_global_motion_features(previous_gray, next_gray, config)
    if len(source) < 4 or len(target) < 4:
        return None, match_count, 0

    transform, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(config.ransac_reprojection_px),
        maxIters=2000,
        confidence=0.995,
    )
    if transform is None or not np.isfinite(transform).all():
        return None, match_count, 0
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    return transform.astype(np.float32), match_count, inlier_count


def acceptable_global_motion(transform: np.ndarray, inlier_count: int, config: GlobalMotionConfig) -> bool:
    if inlier_count < int(config.min_inliers):
        return False
    scale, angle = transform_scale_angle(transform)
    scale_change = abs(scale - 1.0)
    translation = float(np.linalg.norm(transform[:, 2]))
    return (
        translation <= float(config.max_translation_px)
        and scale_change <= float(config.max_scale_change)
        and abs(angle) <= float(config.max_rotation_deg)
    )


def smooth_incremental_transform(transform: np.ndarray, smoothing: float) -> np.ndarray:
    smoothing = float(np.clip(smoothing, 0.0, 0.98))
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return (identity * smoothing + transform.astype(np.float32) * (1.0 - smoothing)).astype(np.float32)


def write_output_frame(writer: cv2.VideoWriter, frame_rgb: np.ndarray) -> None:
    writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def run_global_motion(
    video_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    initial_projected_points: np.ndarray,
    faces: list[list[int]],
    face_colors: list[tuple[int, int, int]],
    render_style: str,
    opacity: float,
    config: GlobalMotionConfig,
    status: StatusCollector,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, previous_bgr = cap.read()
    if not ok or previous_bgr is None:
        cap.release()
        raise RuntimeError(f"Could not read frame {start_frame} from {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    height, width = previous_bgr.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps, 1.0),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    previous_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
    cumulative_transform = np.eye(3, dtype=np.float32)
    frames_written = 0

    try:
        previous_rgb = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2RGB)
        transformed_points = apply_2d_transform(initial_projected_points, homogeneous_to_affine(cumulative_transform))
        write_output_frame(writer, draw_obj_overlay(previous_rgb, transformed_points, faces, face_colors, render_style, opacity))
        frames_written += 1

        for absolute_frame in range(start_frame + 1, end_frame + 1):
            ok, next_bgr = cap.read()
            if not ok or next_bgr is None:
                break

            next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
            transform, match_count, inlier_count = estimate_global_motion_transform(previous_gray, next_gray, config)
            accepted = transform is not None and acceptable_global_motion(transform, inlier_count, config)
            motion_text = "no transform"
            if accepted:
                transform = smooth_incremental_transform(transform, float(config.smoothing))
                scale, angle = transform_scale_angle(transform)
                dx, dy = transform[:, 2]
                motion_text = f"dx={dx:+.1f} dy={dy:+.1f} scale={scale:.3f} rot={angle:+.1f}deg"
                cumulative_transform = affine_to_homogeneous(transform) @ cumulative_transform
            elif transform is not None:
                scale, angle = transform_scale_angle(transform)
                dx, dy = transform[:, 2]
                motion_text = f"rejected dx={dx:+.1f} dy={dy:+.1f} scale={scale:.3f} rot={angle:+.1f}deg"

            next_rgb = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2RGB)
            transformed_points = apply_2d_transform(initial_projected_points, homogeneous_to_affine(cumulative_transform))
            drawn = draw_obj_overlay(next_rgb, transformed_points, faces, face_colors, render_style, opacity)
            write_output_frame(writer, drawn)
            frames_written += 1

            state = "accepted" if accepted else "frozen"
            status.caption(
                f"Frame {absolute_frame} / {end_frame}: {state}, "
                f"{inlier_count}/{match_count} inliers, {motion_text}"
            )
            previous_gray = next_gray
    finally:
        writer.release()
        cap.release()

    if frames_written <= 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("OpenCV Global Motion did not create an output video.")
    return frames_written


def parse_local_paths(raw_value: str) -> list[Path]:
    paths: list[Path] = []
    for raw_path in raw_value.replace("\n", ",").split(","):
        item = raw_path.strip()
        if item:
            paths.append(Path(item).expanduser())
    return paths


def uploaded_files_to_paths(field_name: str, run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for storage in request.files.getlist(field_name):
        if not storage or not storage.filename:
            continue
        filename = secure_filename(storage.filename)
        if not filename:
            continue
        path = run_dir / filename
        storage.save(path)
        paths.append(path)
    return paths


def first_video_frame(video_path: Path, start_frame: int) -> tuple[np.ndarray, float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Could not read frame {start_frame} from {video_path}")
        height, width = frame_bgr.shape[:2]
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), fps, width, height, total_frames
    finally:
        cap.release()


def load_combined_mesh(paths: list[Path]) -> ObjMeshPart:
    material_colors: dict[str, tuple[int, int, int]] = {}
    for path in paths:
        if path.suffix.lower() == ".mtl":
            material_colors.update(parse_mtl_colors(path.read_bytes()))

    obj_paths = [path for path in paths if path.suffix.lower() == ".obj"]
    if not obj_paths:
        raise RuntimeError("Provide at least one OBJ file.")

    meshes = [
        parse_obj_model(
            path.read_bytes(),
            path.name,
            fallback_color=default_obj_color(path.name, index),
            material_colors=material_colors,
        )
        for index, path in enumerate(obj_paths)
    ]
    return combine_obj_meshes(meshes)


def build_projected_overlay(
    mesh: ObjMeshPart,
    frame_width: int,
    frame_height: int,
    settings: dict[str, Any],
) -> np.ndarray:
    normalized = normalize_obj_vertices(mesh.vertices)
    model_points_3d = normalized @ rotation_matrix_xyz(
        settings["rotate_x"],
        settings["rotate_y"],
        settings["rotate_z"],
    ).T
    projected_points = project_normalized_obj_vertices(
        model_points_3d,
        frame_width,
        frame_height,
        settings["center_x"],
        settings["center_y"],
        settings["scale_px"],
    )
    in_frame = (
        (projected_points[:, 0] >= 0)
        & (projected_points[:, 0] < frame_width)
        & (projected_points[:, 1] >= 0)
        & (projected_points[:, 1] < frame_height)
    )
    if not bool(in_frame.any()):
        raise RuntimeError("No projected OBJ anchors are visible in the frame. Move or resize the model.")
    return projected_points


def form_float(name: str, default: float) -> float:
    try:
        return float(request.form.get(name, default))
    except (TypeError, ValueError):
        return default


def form_int(name: str, default: int) -> int:
    try:
        return int(float(request.form.get(name, default)))
    except (TypeError, ValueError):
        return default


def form_settings(frame_width: int, frame_height: int) -> dict[str, Any]:
    return {
        "start_frame": max(0, form_int("start_frame", 0)),
        "end_frame": max(0, form_int("end_frame", 0)),
        "center_x": form_float("center_x", frame_width * 0.5),
        "center_y": form_float("center_y", frame_height * 0.5),
        "scale_px": max(10.0, form_float("scale_px", min(frame_width, frame_height) * DEFAULT_OBJ_SCALE_FRACTION)),
        "rotate_x": form_float("rotate_x", DEFAULT_OBJ_ROTATION[0]),
        "rotate_y": form_float("rotate_y", DEFAULT_OBJ_ROTATION[1]),
        "rotate_z": form_float("rotate_z", DEFAULT_OBJ_ROTATION[2]),
        "opacity": float(np.clip(form_float("opacity", 0.5), 0.0, 1.0)),
        "render_style": request.form.get("render_style", "Wireframe"),
        "max_features": max(200, form_int("max_features", 2000)),
        "min_inliers": max(4, form_int("min_inliers", 30)),
        "ransac_px": max(0.5, form_float("ransac_px", 5.0)),
        "smoothing": float(np.clip(form_float("smoothing", 0.25), 0.0, 0.98)),
        "max_translation_px": max(1.0, form_float("max_translation_px", 80.0)),
        "max_scale_change": max(0.0, form_float("max_scale_change", 0.12)),
        "max_rotation_deg": max(0.1, form_float("max_rotation_deg", 8.0)),
    }


def global_motion_config(settings: dict[str, Any]) -> GlobalMotionConfig:
    return GlobalMotionConfig(
        max_features=settings["max_features"],
        min_inliers=settings["min_inliers"],
        ransac_reprojection_px=settings["ransac_px"],
        smoothing=settings["smoothing"],
        max_translation_px=settings["max_translation_px"],
        max_scale_change=settings["max_scale_change"],
        max_rotation_deg=settings["max_rotation_deg"],
        xy_rotation_source="Disabled",
    )


def path_for_browser(path: Path) -> str:
    relative = path.resolve().relative_to(RUN_DIR.resolve())
    return url_for("runs", filename=str(relative))


def run_or_preview() -> dict[str, Any]:
    run_id = hashlib.sha1(str(time.time_ns()).encode("ascii")).hexdigest()[:12]
    run_dir = RUN_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    uploaded_video_paths = uploaded_files_to_paths("video_upload", run_dir)
    video_paths = parse_local_paths(request.form.get("video_path", ""))
    video_path = video_paths[0] if video_paths else (uploaded_video_paths[0] if uploaded_video_paths else None)
    if video_path is None:
        raise RuntimeError("Provide a video path or upload a video.")
    if not video_path.exists():
        raise RuntimeError(f"Video not found: {video_path}")

    local_model_paths = parse_local_paths(request.form.get("model_paths", ""))
    uploaded_model_paths = uploaded_files_to_paths("model_uploads", run_dir)
    model_paths = local_model_paths + uploaded_model_paths
    missing = [path for path in model_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Model file not found: {missing[0]}")

    start_frame = max(0, form_int("start_frame", 0))
    first_frame_rgb, fps, frame_width, frame_height, total_frames = first_video_frame(video_path, start_frame)
    settings = form_settings(frame_width, frame_height)
    if settings["end_frame"] <= settings["start_frame"]:
        settings["end_frame"] = max(settings["start_frame"], total_frames - 1)
    if total_frames > 0:
        settings["end_frame"] = min(settings["end_frame"], total_frames - 1)

    mesh = load_combined_mesh(model_paths)
    projected_points = build_projected_overlay(mesh, frame_width, frame_height, settings)

    preview_rgb = draw_obj_overlay(
        first_frame_rgb,
        projected_points,
        mesh.faces,
        mesh.face_colors,
        settings["render_style"],
        settings["opacity"],
    )
    preview_path = run_dir / "preview.jpg"
    cv2.imwrite(str(preview_path), cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))

    result: dict[str, Any] = {
        "preview_url": path_for_browser(preview_path),
        "status": ["Preview generated."],
        "settings": settings,
        "video_path": str(video_path),
        "model_paths": "\n".join(str(path) for path in model_paths),
        "frame_info": f"{frame_width}x{frame_height}, {fps:.2f} fps, {total_frames} frames",
    }

    if request.form.get("action") != "run":
        return result

    output_path = run_dir / "global_motion.mp4"
    status = StatusCollector()
    started_at = time.perf_counter()
    frames_written = run_global_motion(
        video_path,
        output_path,
        settings["start_frame"],
        settings["end_frame"],
        projected_points,
        mesh.faces,
        mesh.face_colors,
        settings["render_style"],
        settings["opacity"],
        global_motion_config(settings),
        status,
    )
    elapsed = time.perf_counter() - started_at
    processed_frames = max(1, frames_written)

    result.update(
        {
            "video_url": path_for_browser(output_path),
            "download_url": path_for_browser(output_path),
            "elapsed": f"{elapsed:.2f}s",
            "processed_fps": f"{processed_frames / max(elapsed, 1e-6):.2f} frames/s",
            "status": status.messages[-12:] or ["Run finished."],
        }
    )
    return result


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OpenCV Global Motion Local</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #0f1117; color: #f3f4f6; }
    main { max-width: 1280px; margin: 0 auto; padding: 28px; }
    h1 { margin: 0 0 8px; font-size: 28px; }
    p { color: #a7adbb; }
    form { display: grid; grid-template-columns: 360px 1fr; gap: 20px; align-items: start; }
    section { border: 1px solid #2b303b; border-radius: 8px; padding: 16px; background: #151922; }
    label { display: grid; gap: 7px; margin-bottom: 13px; color: #d8dbe3; font-size: 13px; }
    input, select, textarea, button {
      border: 1px solid #313846; border-radius: 7px; background: #10141d; color: #f3f4f6;
      padding: 10px; font: inherit;
    }
    textarea { min-height: 78px; resize: vertical; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .two { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .actions { display: flex; gap: 10px; margin-top: 8px; }
    button { cursor: pointer; background: #ff475c; border-color: #ff475c; font-weight: 700; }
    button.secondary { background: #242a36; border-color: #3a4252; }
    img, video { width: 100%; border-radius: 8px; border: 1px solid #2b303b; background: #050608; }
    .error { border-color: #ff475c; color: #ffd7dc; background: #2a1218; }
    .stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
    .stat { background: #10141d; border: 1px solid #2b303b; padding: 12px; border-radius: 8px; }
    .stat b { display: block; font-size: 12px; color: #a7adbb; margin-bottom: 5px; }
    pre { white-space: pre-wrap; background: #10141d; border: 1px solid #2b303b; padding: 12px; border-radius: 8px; color: #d8dbe3; }
    @media (max-width: 900px) { form, .grid, .two, .stats { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>OpenCV Global Motion Local</h1>
  <p>Local test UI only for global affine motion + 3D OBJ overlay. No Streamlit, no neural point tracker.</p>

  {% if error %}
    <section class="error"><b>Error</b><br>{{ error }}</section><br>
  {% endif %}

  <form method="post" enctype="multipart/form-data">
    <section>
      <h2>Files</h2>
      <label>Video path on this machine
        <input name="video_path" value="{{ values.video_path or '' }}" placeholder="/path/to/video.mp4">
      </label>
      <label>Or upload video
        <input name="video_upload" type="file" accept="video/*">
      </label>
      <label>OBJ / MTL paths, comma or line separated
        <textarea name="model_paths" placeholder="/path/to/Rein.obj&#10;/path/to/Artere.obj&#10;/path/to/materials.mtl">{{ values.model_paths or '' }}</textarea>
      </label>
      <label>Or upload OBJ / MTL files
        <input name="model_uploads" type="file" accept=".obj,.mtl" multiple>
      </label>

      <h2>Clip</h2>
      <div class="two">
        <label>Start frame <input name="start_frame" type="number" value="{{ values.start_frame or 0 }}"></label>
        <label>End frame, 0 = full video <input name="end_frame" type="number" value="{{ values.end_frame or 0 }}"></label>
      </div>
    </section>

    <section>
      <h2>3D Placement</h2>
      <div class="grid">
        <label>Center X px <input name="center_x" type="number" step="1" value="{{ values.center_x or '' }}"></label>
        <label>Center Y px <input name="center_y" type="number" step="1" value="{{ values.center_y or '' }}"></label>
        <label>Scale px <input name="scale_px" type="number" step="1" value="{{ values.scale_px or '' }}"></label>
        <label>Rotate X deg <input name="rotate_x" type="number" step="1" value="{{ values.rotate_x if values.rotate_x is defined else -40 }}"></label>
        <label>Rotate Y deg <input name="rotate_y" type="number" step="1" value="{{ values.rotate_y if values.rotate_y is defined else 20 }}"></label>
        <label>Rotate Z deg <input name="rotate_z" type="number" step="1" value="{{ values.rotate_z if values.rotate_z is defined else -15 }}"></label>
      </div>
      <div class="grid">
        <label>Render style
          <select name="render_style">
            <option value="Wireframe" {% if values.render_style == "Wireframe" %}selected{% endif %}>Wireframe</option>
            <option value="Filled" {% if values.render_style == "Filled" %}selected{% endif %}>Filled</option>
          </select>
        </label>
        <label>Opacity <input name="opacity" type="number" step="0.05" min="0" max="1" value="{{ values.opacity or 0.5 }}"></label>
      </div>

      <h2>Global Motion</h2>
      <div class="grid">
        <label>ORB features <input name="max_features" type="number" value="{{ values.max_features or 2000 }}"></label>
        <label>Min inliers <input name="min_inliers" type="number" value="{{ values.min_inliers or 30 }}"></label>
        <label>RANSAC px <input name="ransac_px" type="number" step="0.5" value="{{ values.ransac_px or 5.0 }}"></label>
        <label>Smoothing <input name="smoothing" type="number" step="0.05" min="0" max="0.98" value="{{ values.smoothing or 0.25 }}"></label>
        <label>Max translation px/frame <input name="max_translation_px" type="number" step="5" value="{{ values.max_translation_px or 80 }}"></label>
        <label>Max scale change/frame <input name="max_scale_change" type="number" step="0.01" value="{{ values.max_scale_change or 0.12 }}"></label>
        <label>Max rotation deg/frame <input name="max_rotation_deg" type="number" step="1" value="{{ values.max_rotation_deg or 8 }}"></label>
      </div>

      <div class="actions">
        <button class="secondary" name="action" value="preview">Preview placement</button>
        <button name="action" value="run">Run global motion</button>
      </div>
    </section>
  </form>

  {% if result %}
    <br>
    <section>
      <h2>Result</h2>
      <div class="stats">
        <div class="stat"><b>Input</b>{{ result.frame_info }}</div>
        <div class="stat"><b>Elapsed</b>{{ result.elapsed or "preview only" }}</div>
        <div class="stat"><b>Speed</b>{{ result.processed_fps or "-" }}</div>
      </div>
      <h3>First-frame overlay preview</h3>
      <img src="{{ result.preview_url }}">
      {% if result.video_url %}
        <h3>Output video</h3>
        <video controls src="{{ result.video_url }}"></video>
        <p><a href="{{ result.download_url }}">Download output MP4</a></p>
      {% endif %}
      <h3>Status</h3>
      <pre>{% for message in result.status %}{{ message }}
{% endfor %}</pre>
    </section>
  {% endif %}
</main>
</body>
</html>
"""


@app.get("/")
def index() -> str:
    return render_template_string(HTML, error=None, result=None, values={})


@app.post("/")
def submit() -> str:
    try:
        result = run_or_preview()
        return render_template_string(HTML, error=None, result=result, values=result["settings"] | {
            "video_path": result["video_path"],
            "model_paths": result["model_paths"],
        })
    except Exception as error:
        values = dict(request.form)
        return render_template_string(HTML, error=str(error), result=None, values=values), 400


@app.get("/runs/<path:filename>")
def runs(filename: str) -> Response:
    return send_from_directory(RUN_DIR, filename)


@app.get("/health")
def health() -> str:
    return "ok"


if __name__ == "__main__":
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=7860, debug=False)
