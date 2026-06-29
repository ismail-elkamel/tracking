from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from instrument_segmentation.dataset import (
    InstrumentSegmentationDataset,
    discover_samples,
    split_samples,
)
from instrument_segmentation.models import build_model
from instrument_segmentation.models import DEFAULT_MODEL
from instrument_segmentation.export_onnx import export_checkpoint_to_onnx


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    preds = torch.sigmoid(logits) > 0.5
    targets_bool = targets > 0.5
    dims = (1, 2, 3)
    intersection = (preds & targets_bool).sum(dim=dims).float()
    union = (preds | targets_bool).sum(dim=dims).float()
    pred_sum = preds.sum(dim=dims).float()
    target_sum = targets_bool.sum(dim=dims).float()
    iou = ((intersection + eps) / (union + eps)).mean().item()
    dice = ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).mean().item()
    return dice, iou


class VisdomLogger:
    def __init__(self, enabled: bool, env: str, port: int) -> None:
        self.enabled = enabled
        self.viz = None
        self.env = env
        if not enabled:
            return
        try:
            import visdom

            self.viz = visdom.Visdom(port=port, env=env)
            if not self.viz.check_connection(timeout_seconds=3):
                print("Visdom server is not reachable; continuing without Visdom.")
                self.viz = None
                self.enabled = False
        except Exception as error:
            print(f"Could not initialize Visdom: {error}. Continuing without Visdom.")
            self.enabled = False

    def plot(self, name: str, epoch: int, train_value: float, val_value: float) -> None:
        if not self.enabled or self.viz is None:
            return
        self.viz.line(
            X=[epoch],
            Y=[train_value],
            win=name,
            name="train",
            update="append" if epoch > 1 else None,
            opts={"title": name, "xlabel": "epoch"},
        )
        self.viz.line(
            X=[epoch],
            Y=[val_value],
            win=name,
            name="val",
            update="append",
        )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0

    progress = tqdm(loader, leave=False, desc="train" if training else "val")
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = nn.functional.interpolate(
                        logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loss = bce(logits, masks) + dice_loss(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        dice, iou = segmentation_metrics(logits.detach(), masks)
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        batches += 1
        progress.set_postfix(loss=total_loss / batches, dice=total_dice / batches, iou=total_iou / batches)

    return {
        "loss": total_loss / max(1, batches),
        "dice": total_dice / max(1, batches),
        "iou": total_iou / max(1, batches),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an instrument-only segmentation model.")
    parser.add_argument("--data-root", default="data/Instrument segmentation")
    parser.add_argument("--output-dir", default="instrument_segmentation/runs/instrument_model")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-units", nargs="*", default=["U61", "U65"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--visdom", action="store_true")
    parser.add_argument("--visdom-port", type=int, default=8097)
    parser.add_argument("--visdom-env", default="instrument_segmentation")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    samples = discover_samples(args.data_root)
    if not samples:
        raise RuntimeError(f"No image/annotation pairs found under {args.data_root!r}.")

    train_samples, val_samples = split_samples(
        samples,
        val_units=args.val_units,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if not train_samples or not val_samples:
        raise RuntimeError("Train/validation split is empty. Adjust --val-units or --val-ratio.")
    if args.max_train_samples is not None:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_samples = val_samples[: args.max_val_samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print("Only objects with classTitle == 'Instrument' are used as positive mask pixels.")

    train_ds = InstrumentSegmentationDataset(train_samples, image_size=args.image_size, augment=True)
    val_ds = InstrumentSegmentationDataset(val_samples, image_size=args.image_size, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(device.type, enabled=use_amp)
    visdom_logger = VisdomLogger(args.visdom, env=args.visdom_env, port=args.visdom_port)

    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, scaler, use_amp)
        val_metrics = run_epoch(model, val_loader, None, device, scaler, use_amp)
        scheduler.step()

        visdom_logger.plot("loss", epoch, train_metrics["loss"], val_metrics["loss"])
        visdom_logger.plot("dice", epoch, train_metrics["dice"], val_metrics["dice"])
        visdom_logger.plot("iou", epoch, train_metrics["iou"], val_metrics["iou"])

        checkpoint = {
            "epoch": epoch,
            "model": args.model,
            "image_size": args.image_size,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(checkpoint, output_dir / "best.pt")

        elapsed = time.time() - start
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} time={elapsed:.1f}s"
        )

    best_checkpoint = output_dir / "best.pt"
    print(f"Best checkpoint: {best_checkpoint}")
    if args.export_onnx:
        onnx_output = Path(args.onnx_output) if args.onnx_output else output_dir / "best.onnx"
        exported_path = export_checkpoint_to_onnx(best_checkpoint, onnx_output, device_name="cpu")
        print(f"Best ONNX model: {exported_path}")


if __name__ == "__main__":
    main()
