#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


class Sample:
    def __init__(self, image_path: Path, json_path: Path) -> None:
        self.image_path = image_path
        self.json_path = json_path


def discover_samples(root: Path, dataset_name: str = "polished") -> List[Sample]:
    dataset_roots = {
        "polished": root / "DATA" / "Polished_PNGs_withAnnotation",
        "asbuilt": root / "DATA" / "Asbuilt_PNGs_with_Annotation",
    }
    if dataset_name not in dataset_roots:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    datasets = [dataset_roots[dataset_name]]
    samples: List[Sample] = []
    for dataset_root in datasets:
        image_dir = dataset_root / "PNGs"
        json_dir = dataset_root / "JSONs"
        if not image_dir.exists() or not json_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("*.png")):
            json_path = json_dir / f"{image_path.stem}.json"
            if json_path.exists():
                samples.append(Sample(image_path=image_path, json_path=json_path))
    if not samples:
        raise FileNotFoundError(f"No PNG/JSON pairs found under {root}")
    return samples


def load_annotation(json_path: Path) -> Tuple[str, Sequence[dict]]:
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not data:
        raise ValueError(f"Annotation file is empty: {json_path}")

    image_key, image_entry = next(iter(data.items()))
    regions = image_entry.get("regions", {})
    if isinstance(regions, dict):
        region_values = regions.values()
    else:
        region_values = regions
    return image_key, list(region_values)


def build_class_map(samples: Sequence[Sample]) -> Dict[str, int]:
    labels = set()
    for sample in samples:
        _, regions = load_annotation(sample.json_path)
        for region in regions:
            label = region.get("region_attributes", {}).get("label", "").strip()
            if label:
                labels.add(label)
    if not labels:
        raise ValueError("No labels found in annotations.")
    class_map = {"background": 0}
    for idx, label in enumerate(sorted(labels), start=1):
        class_map[label] = idx
    return class_map


def rasterize_mask(
    image_size: Tuple[int, int],
    regions: Sequence[dict],
    class_map: Dict[str, int],
) -> Image.Image:
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for region in regions:
        shape = region.get("shape_attributes", {})
        attrs = region.get("region_attributes", {})
        label = attrs.get("label", "").strip()
        class_id = class_map.get(label)
        if not class_id:
            continue
        xs = shape.get("all_points_x", [])
        ys = shape.get("all_points_y", [])
        if len(xs) < 3 or len(xs) != len(ys):
            continue
        polygon = [(float(x), float(y)) for x, y in zip(xs, ys)]
        draw.polygon(polygon, fill=class_id)

    return mask


class PolygonSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        class_map: Dict[str, int],
        image_size: Optional[int],
    ) -> None:
        self.samples = list(samples)
        self.class_map = class_map
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        _, regions = load_annotation(sample.json_path)

        image = Image.open(sample.image_path).convert("RGB")
        mask = rasterize_mask(image.size, regions, self.class_map)

        if self.image_size is not None:
            target_size = (self.image_size, self.image_size)
            image = image.resize(target_size, Image.BILINEAR)
            mask = mask.resize(target_size, Image.NEAREST)

        width, height = image.size

        image_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        image_tensor = image_tensor.view(height, width, 3)
        image_tensor = image_tensor.permute(2, 0, 1).float().div(255.0)

        mask_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
        mask_tensor = mask_tensor.view(height, width).long()
        return image_tensor, mask_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.down4 = DownBlock(base_channels * 8, base_channels * 16)
        self.up1 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smooth: float = 1.0,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    cardinality = torch.sum(probs + one_hot, dims)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets)
    dice = multiclass_dice_loss(logits, targets, num_classes)
    return ce + dice


def compute_class_weights(dataset_subset, num_classes: int) -> torch.Tensor:
    pixel_counts = torch.ones(num_classes, dtype=torch.float32)
    for image_tensor, mask_tensor in dataset_subset:
        counts = torch.bincount(mask_tensor.view(-1), minlength=num_classes).float()
        pixel_counts += counts

    freqs = pixel_counts / pixel_counts.sum()
    # Use a softer weighting curve than full inverse-frequency so rare classes
    # get help without overwhelming the dominant segmentation signal.
    inv_sqrt_freqs = torch.rsqrt(freqs)
    weights = inv_sqrt_freqs / inv_sqrt_freqs.mean()
    return weights


def mean_iou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = torch.argmax(logits, dim=1)
    scores = []
    for class_id in range(1, num_classes):
        pred_mask = preds == class_id
        true_mask = targets == class_id
        union = (pred_mask | true_mask).sum().item()
        if union == 0:
            continue
        intersection = (pred_mask & true_mask).sum().item()
        scores.append(intersection / union)
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    optimizer: Optional[torch.optim.Optimizer],
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(training):
            logits = model(images)
            ce = F.cross_entropy(logits, masks, weight=class_weights)
            dice = multiclass_dice_loss(logits, masks, num_classes)
            loss = ce + dice
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_iou += mean_iou(logits.detach(), masks, num_classes)
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0
    return total_loss / num_batches, total_iou / num_batches


def save_class_map(class_map: Dict[str, int], output_dir: Path) -> Path:
    path = output_dir / "class_map.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(class_map, fh, indent=2)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net on polygon annotations.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--dataset", choices=["polished", "asbuilt"], default="polished")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--early-stopping-patience", type=int, default=60)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_size <= 0:
        args.image_size = None

    samples = discover_samples(args.project_root, dataset_name=args.dataset)
    class_map = build_class_map(samples)
    save_class_map(class_map, args.output_dir)

    dataset = PolygonSegmentationDataset(
        samples=samples,
        class_map=class_map,
        image_size=args.image_size,
    )

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train-ratio + val-ratio + test-ratio must sum to 1.0")

    dataset_size = len(dataset)
    train_size = max(1, int(dataset_size * args.train_ratio))
    val_size = max(1, int(dataset_size * args.val_ratio))
    test_size = dataset_size - train_size - val_size

    if test_size < 1:
        if train_size > val_size and train_size > 1:
            train_size -= 1
        elif val_size > 1:
            val_size -= 1
        test_size = dataset_size - train_size - val_size

    if min(train_size, val_size, test_size) < 1:
        raise ValueError("Dataset is too small for the requested train/val/test split.")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        in_channels=3,
        num_classes=len(class_map),
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    class_weights = compute_class_weights(train_dataset, len(class_map)).to(device)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = args.output_dir / "best_unet.pt"
    latest_path = args.output_dir / "last_unet.pt"
    history = []

    print(f"Using device: {device}")
    print(f"Found {len(samples)} samples")
    print(f"Classes: {class_map}")
    print(f"Class weights: {class_weights.detach().cpu().tolist()}")
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")
    print(f"Test samples: {test_size}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = run_epoch(
            model,
            train_loader,
            device,
            len(class_map),
            optimizer,
            class_weights=class_weights,
        )
        val_loss, val_iou = run_epoch(
            model,
            val_loader,
            device,
            len(class_map),
            optimizer=None,
            class_weights=class_weights,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_map": class_map,
            "class_weights": class_weights.detach().cpu().tolist(),
            "dataset": args.dataset,
            "image_size": args.image_size,
            "base_channels": args.base_channels,
        }
        torch.save(checkpoint, latest_path)
        if val_loss < (best_val_loss - args.early_stopping_min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_mIoU={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} val_mIoU={val_iou:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mIoU": train_iou,
                "val_loss": val_loss,
                "val_mIoU": val_iou,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }
        )
        with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        with (args.output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_mIoU",
                    "val_loss",
                    "val_mIoU",
                    "best_epoch",
                    "best_val_loss",
                ],
            )
            writer.writeheader()
            writer.writerows(history)

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f})"
            )
            break

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_iou = run_epoch(
        model,
        test_loader,
        device,
        len(class_map),
        optimizer=None,
        class_weights=class_weights,
    )

    print(f"Saved best checkpoint to: {best_path}")
    print(f"Saved latest checkpoint to: {latest_path}")
    print(f"Best-checkpoint test_loss={test_loss:.4f} test_mIoU={test_iou:.4f}")


if __name__ == "__main__":
    main()
