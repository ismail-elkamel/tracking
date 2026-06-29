# Surgical Video Tracker

A small Streamlit prototype for testing tracking and prompt-based video segmentation on surgical videos.

## What It Does

- Upload a video.
- Choose a start frame.
- Add point, rectangle, or polygon annotations.
- Track or segment those annotations from the selected frame to the end of the video.
- Compare multiple models and export a labeled collage MP4.
- Save every test into its own folder under `data/output` and offer a download button.
- Optionally hide a point when a tracker loses it or predicts it outside the frame, then resume it when the tracker finds a valid in-frame position again.
- Automatically creates an OpenCV-compatible MP4 copy when OpenCV cannot decode the uploaded codec directly.
- Allows very large uploads through `.streamlit/config.toml` (`102400` MB).
- Keeps tracker/model code in `tracking_methods.py`; `app.py` handles the Streamlit UI.

## Current Backends

- `OpenCV Lucas-Kanade`: point tracking baseline.
- `CoTracker3 Online`: point tracker loaded from `facebookresearch/co-tracker` through PyTorch Hub.
- `CoTracker3 Offline`: CoTracker3 offline model loaded from PyTorch Hub. The app runs it in configurable chunks to avoid GPU out-of-memory errors; lower `CoTracker Offline chunk frames` or `Neural model max side` if memory is tight.
- `LiteTracker`: point tracker cloned in `external/lite-tracker`; uses `models/scaled_online.pth` by default.
- `SAM2`: generic SAM2.1 video segmentation with `sam2.1_hiera_small.pt`.
- `SurgiSAM2`: fine-tuned surgical SAM2 checkpoint, `Curated400_checkpoint_26.pt`.
- `SAM3`: Meta SAM3/SAM3.1 video segmentation, added as a separate external backend.
- `MedSAM2`: medical video segmentation through `external/MedSAM2/infer_prompts.py`.

MedSAM2 is still used by the app, so it is kept. Endo-TTAP was removed because the public repo we found was a project page, not runnable inference code for this app.

TrackTention is not exposed as a runnable tracker. The paper describes a temporal attention layer that consumes point tracks from trackers such as CoTracker, rather than a standalone point-tracking inference backend for this app.

## SAM2 vs SurgiSAM2

`SAM2` and `SurgiSAM2` are now two separate selectable models so you can compare them directly.

For point annotations, the SAM backends now behave like point trackers: the app follows the clicked points with optical flow and draws dots. SAM segmentation is only used for rectangle and polygon annotations. This avoids the misleading behavior where a single point prompt makes SAM segment an arbitrary region and the displayed point jumps to that mask center.

Use the sidebar checkbox `Hide lost points and resume when visible` if you want point trackers to remove a point from the rendered output whenever it is lost or predicted outside the image. The app keeps its last valid coordinate internally and shows it again only when the tracker returns a valid in-frame point.

`SAM2` forces the generic checkpoint:

```text
external/Surgical-SAM-2/checkpoints/sam2.1_hiera_small.pt
```

`SurgiSAM2` requires the fine-tuned surgical checkpoint:

```text
external/Surgical-SAM-2/checkpoints/Curated400_checkpoint_26.pt
```

The app will not silently fall back from `SurgiSAM2` to generic SAM2. If the checkpoint is missing, the SurgiSAM2 run fails and tells you what to download. This is intentional, because otherwise a comparison between SAM2 and SurgiSAM2 would be fake.

The local adapter file is:

```text
external/Surgical-SAM-2/infer_prompts.py
```

It is only an implementation wrapper used by both `SAM2` and `SurgiSAM2`; the Streamlit UI no longer exposes a model called `Surgical-SAM-2`.

## SAM3

SAM3 is real and is cloned under:

```text
external/SAM3
```

The app uses:

```text
external/SAM3/infer_prompts.py
```

Important: SAM3 needs a separate environment and gated checkpoint access. Meta's SAM3 README says to use Python 3.12+, PyTorch 2.7+, a CUDA GPU, and authenticated Hugging Face access to `facebook/sam3` or `facebook/sam3.1`.

This machine currently uses the existing `track_env` environment for SAM3. It has CUDA-enabled PyTorch and the local SAM3 package installed.

If you need to recreate that setup:

```bash
conda activate track_env
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e external/SAM3 pycocotools psutil
hf auth login
```

Then use the `SAM3` option in the app. The default command uses:

```bash
conda run -n track_env python external/SAM3/infer_prompts.py
```

## GPU Status

Your NVIDIA driver is visible through `nvidia-smi`, but the old `track_env` PyTorch build was failing at CUDA initialization. I replaced the unusual `torch 2.12.0+cu130` build with the stable CUDA 12.8 PyTorch build. Recheck with:

```bash
conda activate track_env
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

If this still prints `False`, the remaining issue is outside the app code and is likely driver/runtime state. Restarting the machine after a driver/CUDA change often clears this specific PyTorch `CUDA unknown error`.

## Sources Checked

- SAM2: https://github.com/facebookresearch/sam2
- SurgiSAM2: https://github.com/Devanish31/SurgiSAM2
- SurgiSAM2 checkpoint page: https://figshare.com/articles/media/SurgiSAM2_Fine-tuning_a_foundational_model_for_surgical_video_anatomy_segmentation_and_detection/28489961
- SAM3: https://github.com/facebookresearch/sam3
- MedSAM2: https://github.com/bowang-lab/MedSAM2

## Run

```bash
conda activate track_env
streamlit run app.py
```

Open the local URL Streamlit prints in the terminal.

To recreate the main app environment:

```bash
conda create -n track_env python=3.11 pip
conda activate track_env
python -m pip install -r requirements.txt
```

## Checkpoints

For immediate generic SAM2 testing:

```bash
cd external/Surgical-SAM-2/checkpoints
bash download_ckpts.sh
```

For SurgiSAM2:

```bash
cd external/Surgical-SAM-2/checkpoints
bash download_surgisam2_finetuned.sh
```

For MedSAM2:

```bash
cd external/MedSAM2
SAM2_BUILD_CUDA=0 python -m pip install -e .
bash download.sh
```

## Outputs

Each tracking or comparison test gets its own timestamped folder under:

```text
data/output
```

Example:

```text
data/output/c0d3345a2054_comparison_f0_20260616_153012/
```

Inside that folder you will find the model videos, SAM prompt JSON files when applicable, the comparison collage when applicable, and:

```text
model_timings.csv
```

The CSV records the timestamp, video, mode, model, device, frame range, annotation counts, status, duration in seconds, output path, and any error message.

Leave live preview off when you only want the downloadable result file and do not need Streamlit to render each frame during inference.
