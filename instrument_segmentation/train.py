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


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    tp = (probs * targets).sum(dim=dims)
    fp = (probs * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - probs) * targets).sum(dim=dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky.mean()


def build_loss(loss_name: str):
    if loss_name == "tversky_focal":
        return lambda logits, targets: tversky_loss(logits, targets) + focal_loss(logits, targets)
    if loss_name == "dice_focal":
        return lambda logits, targets: dice_loss(logits, targets) + focal_loss(logits, targets)
    if loss_name == "dice_bce":
        bce = nn.BCEWithLogitsLoss()
        return lambda logits, targets: dice_loss(logits, targets) + bce(logits, targets)
    raise ValueError("Unknown loss. Use one of: tversky_focal, dice_focal, dice_bce.")


def build_optimizer(
    optimizer_name: str,
    parameters,
    lr: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "radam":
        return torch.optim.RAdam(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError("Unknown optimizer. Use one of: adamw, adam, radam, sgd, rmsprop.")


def confusion_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int, int, int]:
    preds = torch.sigmoid(logits) > 0.5
    targets_bool = targets > 0.5
    tp = int((preds & targets_bool).sum().item())
    fp = int((preds & ~targets_bool).sum().item())
    fn = int((~preds & targets_bool).sum().item())
    tn = int((~preds & ~targets_bool).sum().item())
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-6) -> dict[str, float]:
    return {
        "dice": (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "specificity": (tn + eps) / (tn + fp + eps),
    }


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
    loss_fn,
    phase: str | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    tp = fp = fn = tn = 0
    batches = 0

    progress = tqdm(loader, leave=False, desc=phase or ("train" if training else "val"))
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
                loss = loss_fn(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_tp, batch_fp, batch_fn, batch_tn = confusion_counts(logits.detach(), masks)
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn
        tn += batch_tn
        total_loss += loss.item()
        batches += 1
        metrics = metrics_from_counts(tp, fp, fn, tn)
        progress.set_postfix(
            loss=total_loss / batches,
            dice=metrics["dice"],
            iou=metrics["iou"],
            recall=metrics["recall"],
        )

    metrics = metrics_from_counts(tp, fp, fn, tn)
    metrics["loss"] = total_loss / max(1, batches)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an instrument-only segmentation model.")
    parser.add_argument("--data-root", default="data/Instrument segmentation")
    parser.add_argument("--output-dir", default="instrument_segmentation/runs/instrument_model")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "radam", "sgd", "rmsprop"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--loss",
        choices=["tversky_focal", "dice_focal", "dice_bce"],
        default="tversky_focal",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-units", nargs="*", default=["U61", "U65"])
    parser.add_argument("--test-units", nargs="*", default=["UT8", "UT9"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
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

    train_samples, val_samples, test_samples = split_samples(
        samples,
        val_units=args.val_units,
        test_units=args.test_units,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("Train/validation/test split is empty. Adjust --val-units, --test-units, or ratios.")
    if args.max_train_samples is not None:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_samples = val_samples[: args.max_val_samples]
    if args.max_test_samples is not None:
        test_samples = test_samples[: args.max_test_samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print("Only objects with classTitle == 'Instrument' are used as positive mask pixels.")

    train_ds = InstrumentSegmentationDataset(train_samples, image_size=args.image_size, augment=True)
    val_ds = InstrumentSegmentationDataset(val_samples, image_size=args.image_size, augment=False)
    test_ds = InstrumentSegmentationDataset(test_samples, image_size=args.image_size, augment=False)
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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    optimizer = build_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(device.type, enabled=use_amp)
    visdom_logger = VisdomLogger(args.visdom, env=args.visdom_env, port=args.visdom_port)
    loss_fn = build_loss(args.loss)
    print(f"Model: {args.model}")
    print(f"Loss: {args.loss}")
    print(f"Optimizer: {args.optimizer}")

    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, scaler, use_amp, loss_fn, phase="train")
        val_metrics = run_epoch(model, val_loader, None, device, scaler, use_amp, loss_fn, phase="val")
        scheduler.step()

        visdom_logger.plot("loss", epoch, train_metrics["loss"], val_metrics["loss"])
        visdom_logger.plot("dice", epoch, train_metrics["dice"], val_metrics["dice"])
        visdom_logger.plot("iou", epoch, train_metrics["iou"], val_metrics["iou"])
        visdom_logger.plot("recall", epoch, train_metrics["recall"], val_metrics["recall"])
        visdom_logger.plot("precision", epoch, train_metrics["precision"], val_metrics["precision"])

        checkpoint = {
            "epoch": epoch,
            "model": args.model,
            "loss": args.loss,
            "optimizer_name": args.optimizer,
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
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} time={elapsed:.1f}s"
        )

    best_checkpoint = output_dir / "best.pt"
    print(f"Best checkpoint: {best_checkpoint}")
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = run_epoch(model, test_loader, None, device, scaler, use_amp, loss_fn, phase="test")
    print(
        "test "
        f"loss={test_metrics['loss']:.4f} dice={test_metrics['dice']:.4f} "
        f"iou={test_metrics['iou']:.4f} precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} specificity={test_metrics['specificity']:.4f}"
    )
    if args.export_onnx:
        onnx_output = Path(args.onnx_output) if args.onnx_output else output_dir / "best.onnx"
        exported_path = export_checkpoint_to_onnx(best_checkpoint, onnx_output, device_name="cpu")
        print(f"Best ONNX model: {exported_path}")


if __name__ == "__main__":
    main()
