from __future__ import annotations

import queue
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from global_motion_local import (
    DEFAULT_OBJ_ROTATION,
    DEFAULT_OBJ_SCALE_FRACTION,
    GlobalMotionConfig,
    StatusCollector,
    build_projected_overlay,
    draw_obj_overlay,
    first_video_frame,
    load_combined_mesh,
    run_global_motion,
)


OUTPUT_DIR = Path(tempfile.gettempdir()) / "global_motion_desktop"


class GlobalMotionDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("OpenCV Global Motion 3D Test")
        self.root.geometry("1280x820")
        self.root.minsize(1040, 680)

        self.video_path = tk.StringVar()
        self.model_paths: list[Path] = []
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.output_path: Path | None = None
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self.values: dict[str, tk.Variable] = {
            "start_frame": tk.IntVar(value=0),
            "end_frame": tk.IntVar(value=0),
            "center_x": tk.DoubleVar(value=0.0),
            "center_y": tk.DoubleVar(value=0.0),
            "scale_px": tk.DoubleVar(value=0.0),
            "rotate_x": tk.DoubleVar(value=DEFAULT_OBJ_ROTATION[0]),
            "rotate_y": tk.DoubleVar(value=DEFAULT_OBJ_ROTATION[1]),
            "rotate_z": tk.DoubleVar(value=DEFAULT_OBJ_ROTATION[2]),
            "opacity": tk.DoubleVar(value=0.5),
            "render_style": tk.StringVar(value="Wireframe"),
            "max_features": tk.IntVar(value=2000),
            "min_inliers": tk.IntVar(value=30),
            "ransac_px": tk.DoubleVar(value=5.0),
            "smoothing": tk.DoubleVar(value=0.25),
            "max_translation_px": tk.DoubleVar(value=80.0),
            "max_scale_change": tk.DoubleVar(value=0.12),
            "max_rotation_deg": tk.DoubleVar(value=8.0),
        }

        self._build_ui()
        self._poll_worker_queue()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.root, padding=14)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.columnconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=14)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        self._files_panel(sidebar)
        self._placement_panel(sidebar)
        self._motion_panel(sidebar)
        self._actions_panel(sidebar)

        title = ttk.Label(main, text="Preview / Result", font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.preview_label = ttk.Label(main, anchor="center")
        self.preview_label.grid(row=1, column=0, sticky="nsew")

        self.info_label = ttk.Label(main, text="Select a video and OBJ/MTL files, then click Preview placement.")
        self.info_label.grid(row=2, column=0, sticky="ew", pady=(10, 6))

        self.log_text = tk.Text(main, height=8, wrap="word")
        self.log_text.grid(row=3, column=0, sticky="ew")
        self.log("Ready.")

    def _files_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Files", padding=10)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Video").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.video_path, width=42).grid(row=1, column=0, sticky="ew", pady=(3, 6))
        ttk.Button(frame, text="Choose video", command=self.choose_video).grid(row=2, column=0, sticky="ew")

        ttk.Label(frame, text="OBJ / MTL files").grid(row=3, column=0, sticky="w", pady=(10, 3))
        self.model_list = tk.Listbox(frame, height=5)
        self.model_list.grid(row=4, column=0, sticky="ew")
        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(buttons, text="Add model files", command=self.add_model_files).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="Clear", command=self.clear_model_files).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        clip = ttk.Frame(frame)
        clip.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        clip.columnconfigure((0, 1), weight=1)
        self._entry(clip, "Start frame", "start_frame", 0, 0)
        self._entry(clip, "End frame, 0 full", "end_frame", 0, 1)

    def _placement_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="3D Placement", padding=10)
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure((0, 1), weight=1)

        self._entry(frame, "Center X", "center_x", 0, 0)
        self._entry(frame, "Center Y", "center_y", 0, 1)
        self._entry(frame, "Scale px", "scale_px", 1, 0)
        self._entry(frame, "Opacity", "opacity", 1, 1)
        self._entry(frame, "Rotate X", "rotate_x", 2, 0)
        self._entry(frame, "Rotate Y", "rotate_y", 2, 1)
        self._entry(frame, "Rotate Z", "rotate_z", 3, 0)

        ttk.Label(frame, text="Render style").grid(row=3, column=1, sticky="w")
        ttk.Combobox(
            frame,
            textvariable=self.values["render_style"],
            values=["Wireframe", "Filled"],
            state="readonly",
            width=16,
        ).grid(row=4, column=1, sticky="ew", padx=(5, 0), pady=(2, 7))

    def _motion_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="OpenCV Global Motion", padding=10)
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure((0, 1), weight=1)

        self._entry(frame, "ORB features", "max_features", 0, 0)
        self._entry(frame, "Min inliers", "min_inliers", 0, 1)
        self._entry(frame, "RANSAC px", "ransac_px", 1, 0)
        self._entry(frame, "Smoothing", "smoothing", 1, 1)
        self._entry(frame, "Max trans px", "max_translation_px", 2, 0)
        self._entry(frame, "Max scale/frame", "max_scale_change", 2, 1)
        self._entry(frame, "Max rot deg", "max_rotation_deg", 3, 0)

    def _actions_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=3, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)

        ttk.Button(frame, text="Preview placement", command=self.preview).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.run_button = ttk.Button(frame, text="Run global motion", command=self.run)
        self.run_button.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(frame, text="Open output folder", command=self.open_output_folder).grid(row=2, column=0, sticky="ew")

    def _entry(self, parent: ttk.Frame, label: str, key: str, row: int, column: int) -> None:
        pad = (0, 5) if column == 0 else (5, 0)
        ttk.Label(parent, text=label).grid(row=row * 2, column=column, sticky="w", padx=pad)
        ttk.Entry(parent, textvariable=self.values[key], width=16).grid(
            row=row * 2 + 1,
            column=column,
            sticky="ew",
            padx=pad,
            pady=(2, 7),
        )

    def choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.video_path.set(path)
        self.initialize_frame_defaults()

    def add_model_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Choose OBJ / MTL files",
            filetypes=[("OBJ / MTL", "*.obj *.mtl"), ("All files", "*.*")],
        )
        for raw_path in paths:
            path = Path(raw_path)
            if path not in self.model_paths:
                self.model_paths.append(path)
                self.model_list.insert(tk.END, str(path))

    def clear_model_files(self) -> None:
        self.model_paths.clear()
        self.model_list.delete(0, tk.END)

    def initialize_frame_defaults(self) -> None:
        path = Path(self.video_path.get())
        if not path.exists():
            return
        try:
            frame_rgb, fps, width, height, total_frames = first_video_frame(path, int(self.values["start_frame"].get()))
        except Exception as error:
            self.log(f"Could not inspect video: {error}")
            return
        self.values["center_x"].set(round(width * 0.5, 1))
        self.values["center_y"].set(round(height * 0.5, 1))
        self.values["scale_px"].set(round(min(width, height) * DEFAULT_OBJ_SCALE_FRACTION, 1))
        if int(self.values["end_frame"].get()) <= 0 and total_frames > 0:
            self.values["end_frame"].set(total_frames - 1)
        self.info_label.configure(text=f"Video: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
        del frame_rgb

    def current_settings(self) -> dict[str, object]:
        return {key: value.get() for key, value in self.values.items()}

    def validate_inputs(self) -> tuple[Path, dict[str, object]]:
        video = Path(self.video_path.get()).expanduser()
        if not video.exists():
            raise RuntimeError("Choose a valid video file.")
        if not self.model_paths:
            raise RuntimeError("Choose at least one OBJ file.")
        missing = [path for path in self.model_paths if not path.exists()]
        if missing:
            raise RuntimeError(f"Missing model file: {missing[0]}")
        settings = self.current_settings()
        settings["start_frame"] = max(0, int(settings["start_frame"]))
        settings["end_frame"] = max(int(settings["start_frame"]), int(settings["end_frame"]))
        settings["opacity"] = float(np.clip(float(settings["opacity"]), 0.0, 1.0))
        return video, settings

    def make_preview(self) -> tuple[np.ndarray, np.ndarray, ObjMeshPart, float, int]:
        video, settings = self.validate_inputs()
        frame_rgb, fps, width, height, total_frames = first_video_frame(video, int(settings["start_frame"]))
        if int(settings["end_frame"]) <= int(settings["start_frame"]) and total_frames > 0:
            settings["end_frame"] = total_frames - 1
            self.values["end_frame"].set(total_frames - 1)
        mesh = load_combined_mesh(self.model_paths)
        projected_points = build_projected_overlay(mesh, width, height, settings)
        preview = draw_obj_overlay(
            frame_rgb,
            projected_points,
            mesh.faces,
            mesh.face_colors,
            str(settings["render_style"]),
            float(settings["opacity"]),
        )
        self.info_label.configure(text=f"Video: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
        return preview, projected_points, mesh, fps, total_frames

    def preview(self) -> None:
        try:
            preview, *_ = self.make_preview()
            self.show_image(preview)
            self.log("Preview generated.")
        except Exception as error:
            messagebox.showerror("Preview failed", str(error))
            self.log(f"Preview failed: {error}")

    def run(self) -> None:
        try:
            video, settings = self.validate_inputs()
            preview, projected_points, mesh, fps, total_frames = self.make_preview()
        except Exception as error:
            messagebox.showerror("Run failed", str(error))
            self.log(f"Run failed: {error}")
            return

        self.show_image(preview)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"global_motion_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        self.output_path = output_path
        self.run_button.configure(state="disabled")
        self.log("Running OpenCV Global Motion...")

        def worker() -> None:
            status = StatusCollector()
            config = GlobalMotionConfig(
                max_features=int(settings["max_features"]),
                min_inliers=int(settings["min_inliers"]),
                ransac_reprojection_px=float(settings["ransac_px"]),
                smoothing=float(settings["smoothing"]),
                max_translation_px=float(settings["max_translation_px"]),
                max_scale_change=float(settings["max_scale_change"]),
                max_rotation_deg=float(settings["max_rotation_deg"]),
            )
            started_at = time.perf_counter()
            try:
                frames_written = run_global_motion(
                    video,
                    output_path,
                    int(settings["start_frame"]),
                    min(int(settings["end_frame"]), total_frames - 1 if total_frames > 0 else int(settings["end_frame"])),
                    projected_points,
                    mesh.faces,
                    mesh.face_colors,
                    str(settings["render_style"]),
                    float(settings["opacity"]),
                    config,
                    status,
                )
                elapsed = time.perf_counter() - started_at
                self.worker_queue.put(
                    (
                        "done",
                        {
                            "output": output_path,
                            "elapsed": elapsed,
                            "frames": frames_written,
                            "messages": status.messages[-12:],
                        },
                    )
                )
            except Exception as error:
                self.worker_queue.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "done":
                    result = payload
                    assert isinstance(result, dict)
                    output_path = result["output"]
                    elapsed = float(result["elapsed"])
                    frames = int(result["frames"])
                    self.run_button.configure(state="normal")
                    self.log(f"Finished: {output_path}")
                    self.log(f"Elapsed: {elapsed:.2f}s, speed: {frames / max(elapsed, 1e-6):.2f} frames/s")
                    for message in result["messages"]:
                        self.log(str(message))
                    messagebox.showinfo("Run finished", f"Output:\n{output_path}\n\nElapsed: {elapsed:.2f}s")
                elif kind == "error":
                    self.run_button.configure(state="normal")
                    self.log(f"Run failed: {payload}")
                    messagebox.showerror("Run failed", str(payload))
        except queue.Empty:
            pass
        self.root.after(200, self._poll_worker_queue)

    def show_image(self, image_rgb: np.ndarray) -> None:
        image = Image.fromarray(image_rgb)
        max_width = max(640, self.preview_label.winfo_width() or 900)
        max_height = max(420, self.preview_label.winfo_height() or 620)
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_photo)

    def open_output_folder(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["xdg-open", str(OUTPUT_DIR)])

    def log(self, message: str) -> None:
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)


def main() -> None:
    root = tk.Tk()
    app = GlobalMotionDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
