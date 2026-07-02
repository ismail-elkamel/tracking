# Surgical Video Tracker

A small Streamlit prototype for testing tracking and prompt-based video segmentation on surgical videos.

## What It Does

- Upload a video.
- Choose a start frame.
- Add point, rectangle, or polygon annotations.
- Optionally upload a `.obj` 3D model, place it on the start frame, and generate tracking points from its projected vertices.
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

## External Stuff

Everything that is not pure app code lives under `external/` or `models/`.

- External repos are Git submodules under `external/`.
- Large model checkpoints are not committed to GitHub.
- Local app adapters are expected as `infer_prompts.py` files inside the external repos.

When cloning the project on a new machine, use:

```bash
git clone --recurse-submodules https://github.com/ismail-elkamel/tracking.git
cd tracking
```

If the repo is already cloned, initialize or repair the external folders with:

```bash
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive
```

This fetches:

```text
external/MedSAM2          https://github.com/bowang-lab/MedSAM2.git
external/SAM3             https://github.com/facebookresearch/sam3.git
external/SurgiSAM2        https://github.com/Devanish31/SurgiSAM2.git
external/Surgical-SAM-2   https://github.com/jinlab-imvr/Surgical-SAM-2.git
external/lite-tracker     https://github.com/ImFusionGmbH/lite-tracker.git
```

Large checkpoints are intentionally not committed to GitHub. Download them locally after the submodules are initialized.

Important: the app also calls local adapter scripts named `infer_prompts.py` for SAM2/SurgiSAM2, SAM3, and MedSAM2. Those files are project adapters, not part of the public upstream repos. If they exist on the old laptop, copy or commit them separately; otherwise the external repo code will be present but the Streamlit commands that call `external/*/infer_prompts.py` will still fail.

Streamlit hides these external trackers automatically when their adapter file is missing, so comparison runs do not fail late with a missing-script error.

Expected external adapter files:

```text
external/Surgical-SAM-2/infer_prompts.py
external/SAM3/infer_prompts.py
external/MedSAM2/infer_prompts.py
```

Expected local checkpoint files:

```text
models/scaled_online.pth
external/Surgical-SAM-2/checkpoints/sam2.1_hiera_small.pt
external/Surgical-SAM-2/checkpoints/Curated400_checkpoint_26.pt
external/MedSAM2/checkpoints/
```

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

Adapter commands:

```bash
python external/Surgical-SAM-2/infer_prompts.py --model-profile sam2
python external/Surgical-SAM-2/infer_prompts.py --model-profile surgisam2
```

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

## MedSAM2

The app uses:

```text
external/MedSAM2/infer_prompts.py
```

The adapter converts the app's point/region prompts into the first-frame mask that upstream MedSAM2 video inference expects.

## External Command Placeholders

External commands may use these placeholders:

- `{video}`: OpenCV-compatible input video path.
- `{start_frame}`: selected starting frame.
- `{prompts}`: JSON file containing selected points/regions.
- `{output}`: MP4 path that the external model must create.
- `{device}`: `cuda` or `cpu`.

Prompt JSON shape:

```json
{
  "start_frame": 0,
  "annotations": [
    {
      "label": "circle 1",
      "kind": "point",
      "points_xy": [[120.0, 80.0]]
    },
    {
      "label": "path 2",
      "kind": "region",
      "points_xy": [[10.0, 20.0], [30.0, 40.0]]
    }
  ]
}
```

The external script must write an annotated/result MP4 at `{output}`.

## Instrument Segmentation Training

The folder `instrument_segmentation/` contains a separate training pipeline for a binary segmentation model that predicts only surgical instruments. The dataset source is:

```text
data/Instrument segmentation
```

Only annotation objects with:

```text
classTitle == "Instrument"
```

are used as positive pixels. Kidney, tumor, vessel, spleen, cavity, ROI, and all other labels are ignored and treated as background. Images with no `Instrument` annotation are skipped by default.

Install the extra training dependencies:

```bash
conda activate track_env
python -m pip install -r requirements.txt
python -m pip install "setuptools<82" wheel
python -m pip install --no-build-isolation visdom
```

Start Visdom in one terminal:

```bash
conda activate track_env
python -m visdom.server -port 8097
```

Open:

```text
http://localhost:8097
```

Train the default model, `DeepLabV3+` with an ImageNet-pretrained EfficientNet-B4 encoder:

```bash
conda activate track_env
python -m instrument_segmentation.train \
  --data-root "data/Instrument segmentation" \
  --output-dir instrument_segmentation/runs/instrument_model \
  --model deeplabv3plus_efficientnet_b4 \
  --loss tversky_focal \
  --optimizer adamw \
  --epochs 50 \
  --batch-size 4 \
  --image-size 512 \
  --visdom \
  --visdom-env instrument_segmentation \
  --visdom-images-env instrument_segmentation_images \
  --visdom-test-images 4 \
  --visdom-hard-images 4 \
  --export-onnx
```

The default split is:

```text
train = all units except validation/test units
val   = U61 U65
test  = UT8 UT9
```

Use validation to choose the model/loss/threshold. Use test only for final performance after choosing settings.

If you want to keep images that have no instrument at all as empty masks, add:

```bash
--keep-empty-samples
```

To choose different validation or test videos:

```bash
python -m instrument_segmentation.train --val-units U61 U65 --test-units UT8 UT9 --visdom
```

Available model names:

```text
unet_efficientnet_b4
unetplusplus_efficientnet_b4
deeplabv3plus_efficientnet_b4
torchvision_deeplabv3_resnet50
```

Available losses:

```text
tversky_focal
dice_focal
dice_bce
```

Available optimizers:

```text
adamw
adam
radam
sgd
rmsprop
```

The default `tversky_focal` is chosen for instrument avoidance because it gives more weight to missing instrument pixels. The terminal prints epoch-level `dice`, `iou`, `precision`, and `recall` for validation, then evaluates the best checkpoint once on the test split. For this project, watch `val_recall` closely because low recall means the mask is missing instrument areas; also watch `val_precision` because low precision means too many false instrument pixels.

Visdom uses two environments, so you get two tabs:

- `instrument_segmentation`: training and validation curves.
- `instrument_segmentation_images`: test images, worst IoU cases, and false positives.

After the final test evaluation, the image tab shows a few test prediction windows. Each panel is:

```text
original | ground truth in green | prediction in red
```

- `test_predictions`: first test examples.
- `test_worst_iou`: test examples where prediction is farthest from the ground truth.
- `test_false_positives`: test examples with the highest false-positive rate, where red appears outside green.

Use `--visdom-test-images 0` to disable the regular test previews. Use `--visdom-hard-images 0` to disable the worst-case previews.

Checkpoints are written locally and ignored by Git:

```text
instrument_segmentation/runs/instrument_model/best.pt
instrument_segmentation/runs/instrument_model/last.pt
instrument_segmentation/runs/instrument_model/best.onnx
```

The `runs/` folder is only local output:

- `best.pt`: best PyTorch checkpoint from training.
- `last.pt`: latest PyTorch checkpoint from training.
- `best.onnx`: exported model for integration/inference outside PyTorch.
- `best.onnx.data`: extra ONNX weight data that PyTorch may create for larger models. Keep it next to `best.onnx`; it belongs to the ONNX model.
- `onnx_predictions/`: image overlays from `test_onnx.py`.
- `*_instrument_overlay.mp4`: video with red instrument overlay.
- `*_instrument_mask.mp4`: optional black/white mask video.

If you want a clean folder before a serious run:

```bash
rm -rf instrument_segmentation/runs/instrument_model
```

Preview a trained model on one image:

```bash
python -m instrument_segmentation.predict \
  --checkpoint instrument_segmentation/runs/instrument_model/best.pt \
  --image "data/Instrument segmentation/UT1/img/YOUR_IMAGE.png" \
  --output instrument_segmentation/runs/preview.png
```

Export an existing PyTorch checkpoint to ONNX:

```bash
python -m instrument_segmentation.export_onnx \
  --checkpoint instrument_segmentation/runs/instrument_model/best.pt \
  --output instrument_segmentation/runs/instrument_model/best.onnx
```

Test the ONNX model on a new image or a folder of images:

```bash
python -m instrument_segmentation.test_onnx \
  --onnx instrument_segmentation/runs/instrument_model/best.onnx \
  --input "data/Instrument segmentation/UT1/img" \
  --output-dir instrument_segmentation/runs/onnx_predictions \
  --image-size 512 \
  --threshold 0.5
```

Run ONNX inference directly on a video:

```bash
python -m instrument_segmentation.infer_video \
  --onnx instrument_segmentation/runs/instrument_model/best.onnx \
  --video data/input/video1.mp4 \
  --output instrument_segmentation/runs/video1_instrument_overlay.mp4 \
  --image-size 512 \
  --threshold 0.5
```

To also save a black/white instrument mask video:

```bash
python -m instrument_segmentation.infer_video \
  --onnx instrument_segmentation/runs/instrument_model/best.onnx \
  --video data/input/video1.mp4 \
  --output instrument_segmentation/runs/video1_instrument_overlay.mp4 \
  --save-mask-video instrument_segmentation/runs/video1_instrument_mask.mp4
```

## Instrument Avoidance During Tracking

The Streamlit tracker can use the trained instrument ONNX model to avoid tracking points on surgical tools. This applies to the built-in local trackers: OpenCV Lucas-Kanade, CoTracker3, and LiteTracker.

In the sidebar, enable:

```text
Avoid instruments with ONNX mask
```

Default model path:

```text
instrument_segmentation/runs/instrument_model/best.onnx
```

If PyTorch exported a sidecar file, keep it next to the ONNX file:

```text
instrument_segmentation/runs/instrument_model/best.onnx.data
```

Controls:

- `Instrument mask threshold`: lower values avoid more pixels, higher values are stricter.
- `Instrument avoid margin px`: dilates the instrument mask so points stay a few pixels away from tools.
- `Instrument mask model size`: input size used for ONNX inference, usually the training `image-size`.

When a tracked point lands on the instrument mask, the app hides it and keeps its last valid non-instrument position internally. The point can appear again when the tracker predicts it outside the instrument mask.

## 3D Model Overlay During Tracking

The Streamlit app can load an `.obj` model and project it onto the selected start frame. Use the sidebar upload:

```text
3D model overlay (.obj)
```

After uploading, use the `3D model placement` controls to move, scale, and rotate the model. With `Move/zoom 3D model with mouse` enabled, the blue placement box appears next to the point/region annotation canvas: drag the box to translate the model and resize it to zoom. Rotations stay available as sliders. The combined preview is shown below those controls at the same column width. The app tracks visible projected OBJ points as anchors, estimates a global transform from them, and redraws the complete OBJ geometry in the final video as a 50% opacity cyan overlay. Displayed OBJ anchor dots are reprojected from that global model transform, so they stay attached to the 3D model instead of drifting outside it. Regular tracking points stay visible on top, and `Compare models` still works.

Use `3D overlay transform` to choose how the full OBJ follows the tracked anchors:

- `PnP`: estimates a 3D pose from OBJ vertex coordinates to tracked 2D anchors, then reprojects the complete OBJ. This is the best first choice for keeping the model coherent during zoom or camera/viewpoint changes.
- `Similarity`: older 2D fallback using translation, rotation, and scale. Use it if PnP becomes unstable on a difficult clip.

PnP uses an approximate camera matrix from the video size when no calibration is available. For better registration later, replace that approximation with real laparoscope camera intrinsics. The PnP controls let you tune the pose robustness:

- `PnP reprojection error px`: increase it if good points are rejected because tracking is noisy; decrease it if the model jumps because bad points are accepted.
- `PnP min inliers`: minimum number of good tracked anchors needed before the app accepts the PnP pose. If this is too high, the app falls back to the 2D similarity transform more often.
- `3D model tracking points`: number of visible OBJ anchors sent to the tracker. More points make PnP more stable, but also make tracking slower.
- `Show 3D anchor points in output`: keep it enabled when debugging registration; disable it when you only want the 50% opacity OBJ volume in the final video.

The preview and final video render all OBJ faces instead of only the first faces, so dense kidney meshes should appear as the full projected model.

Current behavior is a lightweight 2D orthographic projection for interactive testing. It is useful for quickly checking whether point tracking can keep a coarse model overlay aligned, but it is not yet a camera-calibrated 3D registration pipeline.

## Tracking Drift Filters

If tracked points get stuck on the image border or jump suddenly, enable:

```text
Reject border/jump drift
```

Controls:

- `Reject points within edge px`: hides points too close to the frame edge.
- `Reject points near content edge px`: hides points too close to the non-black surgical image boundary.
- `Reject point jumps over px`: hides points that move too far from their last valid position in one step.

Default conservative values:

```text
Hide lost points and resume when visible = on
Reject points within edge px = 32
Reject points near content edge px = 48
Reject point jumps over px = 50
```

The point cloud also defaults to drawn rectangle/polygon areas, with a larger grid margin. This avoids filling the full frame with edge points. If valid fast points disappear, increase the jump limit. If bad points still stick to the border, increase the edge margin.

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
git submodule update --init --recursive
```

## Checkpoints

LiteTracker point tracking weights:

```bash
mkdir -p models
python - <<'PY'
from pathlib import Path
import urllib.request

url = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth"
path = Path("models/scaled_online.pth")
urllib.request.urlretrieve(url, path)
print(path)
PY
```

Generic SAM2.1 small checkpoint:

```bash
cd external/Surgical-SAM-2/checkpoints
bash download_ckpts.sh
cd ../../..
```

SurgiSAM2 fine-tuned checkpoint:

Download `Curated400_checkpoint_26.pt` from the SurgiSAM2 checkpoint page listed below, then place it here:

```bash
mkdir -p external/Surgical-SAM-2/checkpoints
# copy or move the downloaded file to:
# external/Surgical-SAM-2/checkpoints/Curated400_checkpoint_26.pt
```

MedSAM2 checkpoints and package install:

```bash
cd external/MedSAM2
SAM2_BUILD_CUDA=0 python -m pip install -e .
bash download.sh
cd ../..
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
