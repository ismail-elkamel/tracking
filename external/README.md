# External Model Adapters

The Streamlit app calls external model repos through small adapter scripts.

## SAM2

UI name: `SAM2`

Command:

```bash
python external/Surgical-SAM-2/infer_prompts.py --model-profile sam2
```

This forces the generic SAM2.1 small checkpoint:

```text
external/Surgical-SAM-2/checkpoints/sam2.1_hiera_small.pt
```

## SurgiSAM2

UI name: `SurgiSAM2`

Command:

```bash
python external/Surgical-SAM-2/infer_prompts.py --model-profile surgisam2
```

This requires the fine-tuned surgical checkpoint:

```text
external/Surgical-SAM-2/checkpoints/Curated400_checkpoint_26.pt
```

No fallback is used for this mode, so comparing `SAM2` vs `SurgiSAM2` means you are really comparing generic vs fine-tuned weights.

## SAM3

UI name: `SAM3`

Command:

```bash
conda run -n sam3 python external/SAM3/infer_prompts.py
```

SAM3 needs a separate `sam3` environment, CUDA, and Hugging Face access to `facebook/sam3` or `facebook/sam3.1`.

## MedSAM2

UI name: `MedSAM2`

Command:

```bash
python external/MedSAM2/infer_prompts.py
```

The adapter converts the app's point/region prompts into the first-frame mask that upstream MedSAM2 video inference expects.

## Command Placeholders

External commands may use these placeholders:

- `{video}`: OpenCV-compatible input video path.
- `{start_frame}`: Selected starting frame.
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
