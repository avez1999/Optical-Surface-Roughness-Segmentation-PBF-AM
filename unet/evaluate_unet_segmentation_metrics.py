#!/usr/bin/env python3
"""Evaluate a vanilla U-Net checkpoint on its reproducible held-out test split."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import random_split


UNET_ROOT = Path("/storage/work/aas6636/R_CNN/unet")
sys.path.insert(0, str(UNET_ROOT))
from export_test_predictions import split_sizes  # noqa: E402
from train_unet import PolygonSegmentationDataset, UNet, build_class_map, discover_samples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate standard segmentation metrics for a U-Net checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=UNET_ROOT.parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summary(values: List[float]) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_map = checkpoint["class_map"]
    dataset_name = checkpoint["dataset"]
    samples = discover_samples(args.project_root, dataset_name=dataset_name)
    if build_class_map(samples) != class_map:
        raise ValueError("Dataset class map does not match checkpoint class map.")

    dataset = PolygonSegmentationDataset(samples, class_map, checkpoint["image_size"])
    _, _, test_dataset = random_split(
        dataset,
        list(split_sizes(len(dataset), 0.7, 0.2, 0.1)),
        generator=torch.Generator().manual_seed(args.seed),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3, len(class_map), checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels = {class_id: class_name for class_name, class_id in class_map.items()}
    totals = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in labels}
    per_image: List[Dict] = []
    total_correct = 0
    total_pixels = 0
    with torch.no_grad():
        for dataset_index in test_dataset.indices:
            image, target = dataset[dataset_index]
            prediction = torch.argmax(model(image.unsqueeze(0).to(device)), dim=1).squeeze(0).cpu()
            pred = prediction.numpy()
            true = target.numpy()
            image_metrics = {}
            for class_id, class_name in labels.items():
                tp = int(np.logical_and(pred == class_id, true == class_id).sum())
                fp = int(np.logical_and(pred == class_id, true != class_id).sum())
                fn = int(np.logical_and(pred != class_id, true == class_id).sum())
                totals[class_id]["tp"] += tp
                totals[class_id]["fp"] += fp
                totals[class_id]["fn"] += fn
                image_metrics[class_name] = {
                    "iou": ratio(tp, tp + fp + fn),
                    "dice": ratio(2 * tp, 2 * tp + fp + fn),
                    "precision": ratio(tp, tp + fp),
                    "recall": ratio(tp, tp + fn),
                }
            per_image.append({
                "image": samples[dataset_index].image_path.name,
                "metrics": image_metrics,
            })
            total_correct += int((pred == true).sum())
            total_pixels += int(true.size)

    per_class = {}
    for class_id, class_name in labels.items():
        tp, fp, fn = (totals[class_id][key] for key in ("tp", "fp", "fn"))
        per_class[class_name] = {
            "iou": ratio(tp, tp + fp + fn),
            "dice": ratio(2 * tp, 2 * tp + fp + fn),
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "support_pixels": tp + fn,
        }
    foreground = [name for name in class_map if name != "background"]
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset": dataset_name,
        "test_image_count": len(per_image),
        "split_seed": args.seed,
        "pixel_accuracy": ratio(total_correct, total_pixels),
        "macro_iou_excluding_background": float(np.mean([per_class[name]["iou"] for name in foreground])),
        "macro_dice_excluding_background": float(np.mean([per_class[name]["dice"] for name in foreground])),
        "macro_precision_excluding_background": float(np.mean([per_class[name]["precision"] for name in foreground])),
        "macro_recall_excluding_background": float(np.mean([per_class[name]["recall"] for name in foreground])),
        "per_class": per_class,
        "per_image_macro_iou_excluding_background": summary([
            float(np.mean([item["metrics"][name]["iou"] for name in foreground])) for item in per_image
        ]),
        "per_image_macro_dice_excluding_background": summary([
            float(np.mean([item["metrics"][name]["dice"] for name in foreground])) for item in per_image
        ]),
        "per_image": per_image,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: report[key] for key in report if key not in {"per_class", "per_image"}}, indent=2))


if __name__ == "__main__":
    main()
