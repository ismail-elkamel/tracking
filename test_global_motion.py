from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone OpenCV global motion test. It estimates frame-to-frame "
            "2D affine motion with ORB matches + RANSAC and writes an annotated "
            "video plus a CSV log."
        )
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", default="global_motion_test.mp4", help="Output annotated MP4 path.")
    parser.add_argument("--csv", default="global_motion_test.csv", help="Output CSV log path.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to process.")
    parser.add_argument("--end-frame", type=int, default=-1, help="Last frame to process, inclusive. -1 means video end.")
    parser.add_argument("--max-features", type=int, default=2000, help="ORB features per frame.")
    parser.add_argument("--min-inliers", type=int, default=30, help="Minimum RANSAC inliers to accept motion.")
    parser.add_argument("--ransac-px", type=float, default=5.0, help="RANSAC reprojection threshold in pixels.")
    parser.add_argument("--smoothing", type=float, default=0.25, help="Incremental affine smoothing, 0 raw to 0.98 frozen.")
    parser.add_argument("--max-translation", type=float, default=80.0, help="Max accepted translation per frame in pixels.")
    parser.add_argument("--max-scale-change", type=float, default=0.12, help="Max accepted scale delta per frame.")
    parser.add_argument("--max-rotation", type=float, default=8.0, help="Max accepted in-plane rotation per frame in degrees.")
    parser.add_argument("--draw-matches", action="store_true", help="Draw accepted ORB target points on the video.")
    return parser.parse_args()


def affine_to_homogeneous(transform: np.ndarray) -> np.ndarray:
    homogeneous = np.eye(3, dtype=np.float32)
    homogeneous[:2, :] = transform.astype(np.float32)
    return homogeneous


def homogeneous_to_affine(transform: np.ndarray) -> np.ndarray:
    return transform[:2, :].astype(np.float32)


def apply_affine(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1), dtype=np.float32)
    homogeneous = np.hstack([points.astype(np.float32), ones])
    return (homogeneous @ transform.T).astype(np.float32)


def transform_scale_angle(transform: np.ndarray) -> tuple[float, float]:
    linear = transform[:, :2].astype(np.float32)
    scale_x = float(np.linalg.norm(linear[:, 0]))
    scale_y = float(np.linalg.norm(linear[:, 1]))
    scale = (scale_x + scale_y) * 0.5
    angle = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    return scale, angle


def smooth_incremental_transform(transform: np.ndarray, smoothing: float) -> np.ndarray:
    smoothing = float(np.clip(smoothing, 0.0, 0.98))
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return (identity * smoothing + transform.astype(np.float32) * (1.0 - smoothing)).astype(np.float32)


def match_orb_points(
    previous_gray: np.ndarray,
    next_gray: np.ndarray,
    max_features: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    orb = cv2.ORB_create(
        nfeatures=max(200, int(max_features)),
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=12,
    )
    previous_keypoints, previous_desc = orb.detectAndCompute(previous_gray, None)
    next_keypoints, next_desc = orb.detectAndCompute(next_gray, None)
    if previous_desc is None or next_desc is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, 0
    if len(previous_keypoints) < 4 or len(next_keypoints) < 4:
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


def estimate_affine_motion(
    source: np.ndarray,
    target: np.ndarray,
    match_count: int,
    ransac_px: float,
) -> tuple[np.ndarray | None, np.ndarray, int]:
    if len(source) < 4 or len(target) < 4:
        return None, np.zeros(match_count, dtype=bool), 0

    transform, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_px),
        maxIters=2000,
        confidence=0.995,
    )
    if transform is None or not np.isfinite(transform).all():
        return None, np.zeros(len(source), dtype=bool), 0

    inlier_mask = inliers.reshape(-1).astype(bool) if inliers is not None else np.ones(len(source), dtype=bool)
    return transform.astype(np.float32), inlier_mask, int(inlier_mask.sum())


def acceptable_motion(
    transform: np.ndarray,
    inlier_count: int,
    min_inliers: int,
    max_translation: float,
    max_scale_change: float,
    max_rotation: float,
) -> bool:
    if inlier_count < int(min_inliers):
        return False
    scale, angle = transform_scale_angle(transform)
    translation = float(np.linalg.norm(transform[:, 2]))
    return (
        translation <= float(max_translation)
        and abs(scale - 1.0) <= float(max_scale_change)
        and abs(angle) <= float(max_rotation)
    )


def draw_polyline(output: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    rounded = np.round(points).astype(np.int32)
    cv2.polylines(output, [rounded], True, color, thickness, lineType=cv2.LINE_AA)
    for point in rounded:
        cv2.circle(output, tuple(point), 5, color, -1, lineType=cv2.LINE_AA)


def draw_motion_overlay(
    frame: np.ndarray,
    cumulative_transform: np.ndarray,
    initial_box: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    inlier_mask: np.ndarray,
    draw_matches: bool,
    lines: list[str],
) -> np.ndarray:
    output = frame.copy()
    transformed_box = apply_affine(initial_box, homogeneous_to_affine(cumulative_transform))
    draw_polyline(output, transformed_box, (60, 220, 255), 3)

    center = transformed_box.mean(axis=0)
    cv2.circle(output, tuple(np.round(center).astype(int)), 7, (255, 72, 92), -1, lineType=cv2.LINE_AA)

    if draw_matches and len(target_points):
        inlier_points = target_points[inlier_mask[: len(target_points)]]
        for point in np.round(inlier_points[:: max(1, len(inlier_points) // 120)]).astype(np.int32):
            cv2.circle(output, tuple(point), 2, (20, 184, 124), -1, lineType=cv2.LINE_AA)

    y = 28
    for line in lines:
        cv2.putText(output, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return output


def main() -> None:
    args = parse_args()
    global np
    import numpy as np

    global cv2
    import cv2

    video_path = Path(args.video)
    output_path = Path(args.output)
    csv_path = Path(args.csv)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    start_frame = max(0, int(args.start_frame))
    end_frame = int(args.end_frame)
    if end_frame < 0 and total_frames > 0:
        end_frame = total_frames - 1
    if total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)
    if end_frame < start_frame:
        raise RuntimeError(f"Invalid frame interval: {start_frame} to {end_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, previous_frame = cap.read()
    if not ok or previous_frame is None:
        raise RuntimeError(f"Could not read start frame {start_frame}")

    height, width = previous_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps, 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    margin_x = width * 0.18
    margin_y = height * 0.18
    initial_box = np.array(
        [
            [margin_x, margin_y],
            [width - margin_x, margin_y],
            [width - margin_x, height - margin_y],
            [margin_x, height - margin_y],
        ],
        dtype=np.float32,
    )

    cumulative_transform = np.eye(3, dtype=np.float32)
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    rows: list[dict[str, str | int | float]] = []
    frames_written = 0
    start_time = time.perf_counter()

    try:
        first_overlay = draw_motion_overlay(
            previous_frame,
            cumulative_transform,
            initial_box,
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=bool),
            False,
            [
                f"OpenCV Global Motion test frame {start_frame}/{end_frame}",
                "cyan box = cumulative affine motion from first frame",
            ],
        )
        writer.write(first_overlay)
        frames_written += 1

        for frame_index in range(start_frame + 1, end_frame + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            source, target, match_count = match_orb_points(previous_gray, next_gray, args.max_features)
            transform, inlier_mask, inlier_count = estimate_affine_motion(source, target, match_count, args.ransac_px)

            accepted = False
            raw_dx = raw_dy = raw_scale = raw_angle = np.nan
            if transform is not None:
                raw_scale, raw_angle = transform_scale_angle(transform)
                raw_dx, raw_dy = float(transform[0, 2]), float(transform[1, 2])
                accepted = acceptable_motion(
                    transform,
                    inlier_count,
                    args.min_inliers,
                    args.max_translation,
                    args.max_scale_change,
                    args.max_rotation,
                )
                if accepted:
                    transform = smooth_incremental_transform(transform, args.smoothing)
                    cumulative_transform = affine_to_homogeneous(transform) @ cumulative_transform

            cumulative_affine = homogeneous_to_affine(cumulative_transform)
            cumulative_scale, cumulative_angle = transform_scale_angle(cumulative_affine)
            cumulative_dx, cumulative_dy = float(cumulative_affine[0, 2]), float(cumulative_affine[1, 2])

            state = "accepted" if accepted else "frozen"
            lines = [
                f"frame {frame_index}/{end_frame} | {state}",
                f"inliers {inlier_count}/{match_count} | dx {raw_dx:+.1f} dy {raw_dy:+.1f}",
                f"scale {raw_scale:.3f} | rotZ {raw_angle:+.1f} deg",
                f"cumulative dx {cumulative_dx:+.1f} dy {cumulative_dy:+.1f} scale {cumulative_scale:.3f} rotZ {cumulative_angle:+.1f}",
            ]
            overlay = draw_motion_overlay(
                frame,
                cumulative_transform,
                initial_box,
                source,
                target,
                inlier_mask,
                bool(args.draw_matches),
                lines,
            )
            writer.write(overlay)
            frames_written += 1

            rows.append(
                {
                    "frame": frame_index,
                    "accepted": int(accepted),
                    "matches": match_count,
                    "inliers": inlier_count,
                    "dx": raw_dx,
                    "dy": raw_dy,
                    "scale": raw_scale,
                    "rotation_z_deg": raw_angle,
                    "cumulative_dx": cumulative_dx,
                    "cumulative_dy": cumulative_dy,
                    "cumulative_scale": cumulative_scale,
                    "cumulative_rotation_z_deg": cumulative_angle,
                }
            )
            previous_gray = next_gray
    finally:
        cap.release()
        writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "frame",
            "accepted",
            "matches",
            "inliers",
            "dx",
            "dy",
            "scale",
            "rotation_z_deg",
            "cumulative_dx",
            "cumulative_dy",
            "cumulative_scale",
            "cumulative_rotation_z_deg",
        ]
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    duration = time.perf_counter() - start_time
    print(f"Wrote {frames_written} frames to {output_path}")
    print(f"Wrote motion CSV to {csv_path}")
    print(f"Processed in {duration:.2f}s")


if __name__ == "__main__":
    main()
