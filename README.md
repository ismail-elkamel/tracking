# Surgical Video Tracker

Streamlit prototype for testing surgical video tracking, 3D overlays, and global camera motion.

The current focus of the repo is:

- comparing point trackers on surgical clips;
- placing `.obj` anatomy models on a start frame;
- moving the 3D overlay with OpenCV global image motion;
- optionally avoiding instruments with a local ONNX instrument mask;
- exporting annotated MP4 outputs and timing CSV files.

## Repository Status

The repo has been cleaned to keep only the backends that are still exposed by the app.

Kept:

- `OpenCV Lucas-Kanade`
- `OpenCV Global Motion`
- `CoTracker3 Online`
- `CoTracker3 Offline`
- `LiteTracker`
- `TAPIR`
- `BootsTAPIR`
- `SAM3`
- `MedSAM2`
- `instrument_segmentation/`
- `test_global_motion.py`

Removed from the GitHub repo:

- `kidney_segmentation/`, because it is not used by the current app;
- `external/SurgiSAM2`;
- `external/Surgical-SAM-2`;
- old SAM2/SurgiSAM2 README instructions.

Large local files are ignored by Git:

- `data/`
- `models/`
- `*.pt`, `*.pth`, `*.onnx`
- `external/**/checkpoints/`
- segmentation run folders

## Install

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/ismail-elkamel/tracking.git
cd tracking
```

If already cloned:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Install Python dependencies in your environment:

```bash
python -m pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## App Workflows

The home page has two main workflows.

### Compare Models

Runs two or more selected trackers and creates a labeled comparison collage.

Typical use:

1. Upload a video.
2. Select `Compare models`.
3. Draw points, polygons, or add a 3D model.
4. Select trackers to compare.
5. Click `Compare`.

### OpenCV Global Motion

Runs only the OpenCV global image-motion pipeline. This is the preferred workflow for testing 3D overlay motion without neural point tracking.

Typical use:

1. Upload a video.
2. Select `OpenCV Global Motion`.
3. Upload and place a 3D `.obj` model.
4. Keep `3D anchor source` as `Image motion (no points)`.
5. Tune global motion settings.
6. Click `Track`.

## OpenCV Global Motion

OpenCV Global Motion estimates a 2D transform between each pair of consecutive frames:

```text
frame t -> frame t+1
```

The implementation is in:

```text
tracking_methods.py
```

The core method is:

```text
ORB feature detection
        ↓
Brute-force Hamming matching
        ↓
Lowe ratio filtering
        ↓
cv2.estimateAffinePartial2D with RANSAC
        ↓
accept/reject motion based on safety thresholds
        ↓
smooth accepted transform
        ↓
accumulate transform over time
        ↓
apply cumulative transform to 3D overlay anchors/model
```

It estimates:

- X/Y image translation;
- zoom / scale;
- in-plane rotation, which corresponds visually to image `Z` rotation.

It does not directly estimate true 3D pitch/yaw:

- 3D rotation X;
- 3D rotation Y.

Those rotations need additional assumptions. The app includes experimental X/Y sources:

- `Manual keyframes`
- `Homography X/Y`
- `Homography X/Y + manual keyframes`

The homography modes are experimental because surgical video often violates the assumptions of a stable planar scene:

- deformable organs;
- moving instruments;
- specular highlights;
- non-rigid tissue motion;
- depth changes.

So the recommended stable setup is:

```text
OpenCV Global Motion for translation / zoom / Z rotation
Manual X/Y keyframes only when the camera tilt visibly changes
```

## Global Motion Settings

Good first values:

```text
OpenCV frame step: 1
Global motion ORB features: 2000
Global motion min inliers: 30
Global motion RANSAC px: 5
Global motion smoothing: 0.25
Max translation/frame px: 80
Max scale change/frame: 0.12
Max rotation/frame deg: 8
```

Meaning:

- `ORB features`: maximum feature points detected per frame.
- `min inliers`: minimum RANSAC-agreeing matches required to accept the transform.
- `RANSAC px`: reprojection tolerance in pixels.
- `smoothing`: blends each accepted frame-to-frame transform with identity; higher is more stable but slower to follow real motion.
- `max translation/frame px`: rejects sudden large jumps.
- `max scale change/frame`: rejects sudden zoom spikes.
- `max rotation/frame deg`: rejects sudden in-plane rotation spikes.

## Experimental Homography X/Y

Homography is used only to estimate extra X/Y tilt for the 3D overlay. Translation, zoom, and Z rotation still come from OpenCV Global Motion.

Available point sources:

```text
ORB matches
Central 4 points
```

`Central 4 points` automatically selects four visual points near the center of the image. If instrument avoidance is enabled, it avoids instrument-mask pixels when selecting and validating those points.

Useful settings:

```text
Homography point source: Central 4 points
Central point box size: 0.35
Central point search radius px: 80
Homography X/Y smoothing: 0.85
Max X/Y change/frame deg: 4
Max total X/Y deg: 35
```

If homography is unstable, turn it off and use manual X/Y keyframes.

## Standalone Global Motion Test

For dev-team debugging without Streamlit, use:

```text
test_global_motion.py
```

Example:

```bash
python test_global_motion.py \
  --video data/input/example.mp4 \
  --output global_motion_test.mp4 \
  --csv global_motion_test.csv \
  --draw-matches
```

It writes:

- an annotated MP4 with the accumulated affine motion;
- a CSV with per-frame matches, inliers, translation, scale, and Z rotation.

This script uses only:

- OpenCV;
- NumPy.

It does not use Streamlit, CoTracker, TAPIR, SAM3, MedSAM2, or the 3D overlay code.

## Instrument Segmentation

`instrument_segmentation/` contains the training and export code for the optional instrument-avoidance mask.

The app expects an ONNX model at:

```text
instrument_segmentation/runs/instrument_model/best.onnx
```

This ONNX file is ignored by Git because it is a model artifact.

When enabled in the app, the mask can:

- hide overlay pixels behind instruments;
- prevent some generated/tracked points from being accepted on instrument pixels;
- help the experimental central homography points avoid instruments.

## External Backends

Tracked submodules:

```text
external/MedSAM2       https://github.com/bowang-lab/MedSAM2.git
external/SAM3          https://github.com/facebookresearch/sam3.git
external/lite-tracker  https://github.com/ImFusionGmbH/lite-tracker.git
```

Local adapter scripts:

```text
external/TAPIR/infer_prompts.py
external/SAM3/infer_prompts.py
external/MedSAM2/infer_prompts.py
```

If an external adapter is missing, Streamlit hides that backend instead of failing late.

## Notes For Dev Team

The most important files for the current global-motion work are:

```text
app.py
tracking_methods.py
test_global_motion.py
README.md
```

For debugging only the OpenCV motion estimator, start with `test_global_motion.py`.

For debugging how that motion affects the 3D overlay, use the Streamlit `OpenCV Global Motion` workflow.
