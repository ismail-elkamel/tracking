from __future__ import annotations

import base64
import json
import random
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as F


INSTRUMENT_LABEL = "instrument"


@dataclass(frozen=True)
class SegmentationSample:
    image_path: Path
    annotation_path: Path
    unit: str


def discover_samples(data_root: str | Path) -> list[SegmentationSample]:
    root = Path(data_root)
    samples: list[SegmentationSample] = []
    for unit_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        ann_dir = unit_dir / "ann"
        img_dir = unit_dir / "img"
        if not ann_dir.exists() or not img_dir.exists():
            continue
        for annotation_path in sorted(ann_dir.glob("*.json")):
            image_name = annotation_path.name.removesuffix(".json")
            image_path = img_dir / image_name
            if image_path.exists():
                samples.append(
                    SegmentationSample(
                        image_path=image_path,
                        annotation_path=annotation_path,
                        unit=unit_dir.name,
                    )
                )
    return samples


def split_samples(
    samples: list[SegmentationSample],
    val_units: list[str] | None = None,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[SegmentationSample], list[SegmentationSample]]:
    if val_units:
        val_set = set(val_units)
        train = [sample for sample in samples if sample.unit not in val_set]
        val = [sample for sample in samples if sample.unit in val_set]
        return train, val

    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    return shuffled[val_count:], shuffled[:val_count]


def decode_supervisely_bitmap(bitmap: dict, height: int, width: int) -> np.ndarray:
    compressed = base64.b64decode(bitmap["data"])
    png_bytes = zlib.decompress(compressed)
    bitmap_image = Image.open(BytesIO(png_bytes))
    bitmap_array = np.asarray(bitmap_image)

    if bitmap_array.ndim == 3:
        if bitmap_array.shape[2] == 4:
            local_mask = bitmap_array[:, :, 3] > 0
        else:
            local_mask = bitmap_array.max(axis=2) > 0
    else:
        local_mask = bitmap_array > 0

    origin_x, origin_y = bitmap.get("origin", [0, 0])
    origin_x = int(origin_x)
    origin_y = int(origin_y)

    mask = np.zeros((height, width), dtype=bool)
    local_h, local_w = local_mask.shape[:2]
    x0 = max(0, origin_x)
    y0 = max(0, origin_y)
    x1 = min(width, origin_x + local_w)
    y1 = min(height, origin_y + local_h)
    if x1 <= x0 or y1 <= y0:
        return mask

    src_x0 = x0 - origin_x
    src_y0 = y0 - origin_y
    src_x1 = src_x0 + (x1 - x0)
    src_y1 = src_y0 + (y1 - y0)
    mask[y0:y1, x0:x1] = local_mask[src_y0:src_y1, src_x0:src_x1]
    return mask


def load_instrument_mask(annotation_path: str | Path) -> Image.Image:
    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    height = int(payload["size"]["height"])
    width = int(payload["size"]["width"])
    merged = np.zeros((height, width), dtype=bool)

    for obj in payload.get("objects", []):
        if obj.get("classTitle", "").strip().lower() != INSTRUMENT_LABEL:
            continue
        bitmap = obj.get("bitmap")
        if bitmap:
            merged |= decode_supervisely_bitmap(bitmap, height=height, width=width)

    return Image.fromarray((merged.astype(np.uint8) * 255), mode="L")


class InstrumentSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: list[SegmentationSample],
        image_size: int = 512,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.color_jitter = ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask = load_instrument_mask(sample.annotation_path)

        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if self.augment:
            image = self.color_jitter(image)

        image = F.resize(image, [self.image_size, self.image_size], antialias=True)
        mask = F.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        mask_tensor = (F.to_tensor(mask) > 0.5).float()
        return image_tensor, mask_tensor
