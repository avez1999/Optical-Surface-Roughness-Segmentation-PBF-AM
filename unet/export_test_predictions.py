#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import random_split

from train_unet import (
    PolygonSegmentationDataset,
    UNet,
    build_class_map,
    discover_samples,
)


COLOR_PALETTE = {
    0: (0, 0, 0),
    1: (230, 57, 70),
    2: (29, 161, 242),
    3: (46, 204, 113),
    4: (241, 196, 15),
    5: (155, 89, 182),
    6: (230, 126, 34),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export test-set predictions and metrics.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "outputs" / "best_unet.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def split_sizes(dataset_size: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    train_size = max(1, int(dataset_size * train_ratio))
    val_size = max(1, int(dataset_size * val_ratio))
    test_size = dataset_size - train_size - val_size
    if test_size < 1:
        if train_size > val_size and train_size > 1:
            train_size -= 1
        elif val_size > 1:
            val_size -= 1
        test_size = dataset_size - train_size - val_size
    if min(train_size, val_size, test_size) < 1:
        raise ValueError("Dataset is too small for the requested train/val/test split.")
    return train_size, val_size, test_size


def invert_class_map(class_map: Dict[str, int]) -> Dict[int, str]:
    return {value: key for key, value in class_map.items()}


def overlay_mask(image: Image.Image, mask: np.ndarray, class_map_inv: Dict[int, str]) -> Image.Image:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id in sorted(class_map_inv):
        if class_id == 0:
            continue
        color = COLOR_PALETTE.get(class_id, (255, 255, 255))
        color_mask[mask == class_id] = color
    base = image.convert("RGB")
    overlay = Image.fromarray(color_mask, mode="RGB")
    return Image.blend(base, overlay, alpha=0.4)


def make_triptych(original: Image.Image, gt_overlay: Image.Image, pred_overlay: Image.Image) -> Image.Image:
    width, height = original.size
    canvas = Image.new("RGB", (width * 3, height), (255, 255, 255))
    canvas.paste(original.convert("RGB"), (0, 0))
    canvas.paste(gt_overlay, (width, 0))
    canvas.paste(pred_overlay, (width * 2, 0))
    return canvas


def present_labels(mask: np.ndarray, class_map_inv: Dict[int, str]) -> List[str]:
    labels = []
    for class_id in sorted(np.unique(mask).tolist()):
        if class_id == 0:
            continue
        labels.append(class_map_inv[class_id])
    return labels


def compute_per_class_iou(pred_mask: np.ndarray, true_mask: np.ndarray, class_map_inv: Dict[int, str]) -> Dict[str, float]:
    results = {}
    for class_id, class_name in sorted(class_map_inv.items()):
        if class_id == 0:
            continue
        pred_class = pred_mask == class_id
        true_class = true_mask == class_id
        union = np.logical_or(pred_class, true_class).sum()
        if union == 0:
            continue
        intersection = np.logical_and(pred_class, true_class).sum()
        results[class_name] = float(intersection / union)
    return results


def mean_non_background_iou(per_class_iou: Dict[str, float]) -> float:
    if not per_class_iou:
        return 0.0
    return float(sum(per_class_iou.values()) / len(per_class_iou))


def label_metrics(pred_labels: Sequence[str], true_labels: Sequence[str]) -> Tuple[bool, float, float, float]:
    pred_set = set(pred_labels)
    true_set = set(true_labels)
    exact = pred_set == true_set
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 1.0
    recall = tp / len(true_set) if true_set else 1.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return exact, float(precision), float(recall), float(f1)


def main() -> None:
    args = parse_args()
    export_dir = args.output_dir / "test_predictions"
    images_dir = export_dir / "images"
    json_dir = export_dir / "json"
    images_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_map = checkpoint["class_map"]
    dataset_name = checkpoint.get("dataset", "polished")
    image_size = checkpoint["image_size"]
    base_channels = checkpoint["base_channels"]
    class_map_inv = invert_class_map(class_map)

    samples = discover_samples(args.project_root, dataset_name=dataset_name)
    built_class_map = build_class_map(samples)
    if built_class_map != class_map:
        raise ValueError("Current dataset labels do not match checkpoint class map.")

    dataset = PolygonSegmentationDataset(samples=samples, class_map=class_map, image_size=image_size)
    generator = torch.Generator().manual_seed(args.seed)
    sizes = split_sizes(len(dataset), args.train_ratio, args.val_ratio, args.test_ratio)
    _, _, test_dataset = random_split(dataset, list(sizes), generator=generator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=len(class_map), base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    summary_items = []
    class_iou_totals = {class_name: [] for class_id, class_name in class_map_inv.items() if class_id != 0}

    with torch.no_grad():
        for dataset_index in test_dataset.indices:
            sample = dataset.samples[dataset_index]
            image_tensor, true_mask_tensor = dataset[dataset_index]
            logits = model(image_tensor.unsqueeze(0).to(device))
            pred_mask_tensor = torch.argmax(logits, dim=1).squeeze(0).cpu()

            true_mask = np.array(true_mask_tensor.tolist(), dtype=np.int64)
            pred_mask = np.array(pred_mask_tensor.tolist(), dtype=np.int64)

            base_image = Image.open(sample.image_path).convert("RGB")
            if image_size is not None:
                base_image = base_image.resize((image_size, image_size), Image.BILINEAR)
            gt_overlay = overlay_mask(base_image, true_mask, class_map_inv)
            pred_overlay = overlay_mask(base_image, pred_mask, class_map_inv)
            triptych = make_triptych(base_image, gt_overlay, pred_overlay)

            stem = sample.image_path.stem
            image_output_path = images_dir / f"{stem}_triptych.png"
            triptych.save(image_output_path)

            gt_labels = present_labels(true_mask, class_map_inv)
            pred_labels = present_labels(pred_mask, class_map_inv)
            exact_match, precision, recall, f1 = label_metrics(pred_labels, gt_labels)
            per_class_iou = compute_per_class_iou(pred_mask, true_mask, class_map_inv)
            for class_name, iou in per_class_iou.items():
                class_iou_totals[class_name].append(iou)

            pixel_accuracy = float((pred_mask == true_mask).mean())
            sample_summary = {
                "image_name": sample.image_path.name,
                "annotation_name": sample.json_path.name,
                "prediction_overlay_image": image_output_path.name,
                "ground_truth_labels": gt_labels,
                "predicted_labels": pred_labels,
                "exact_label_match": exact_match,
                "label_precision": precision,
                "label_recall": recall,
                "label_f1": f1,
                "pixel_accuracy": pixel_accuracy,
                "mean_iou_non_background": mean_non_background_iou(per_class_iou),
                "per_class_iou": per_class_iou,
            }
            summary_items.append(sample_summary)

            json_output_path = json_dir / f"{stem}_prediction.json"
            with json_output_path.open("w", encoding="utf-8") as fh:
                json.dump(sample_summary, fh, indent=2)

    exact_matches = [1.0 if item["exact_label_match"] else 0.0 for item in summary_items]
    label_precisions = [item["label_precision"] for item in summary_items]
    label_recalls = [item["label_recall"] for item in summary_items]
    label_f1s = [item["label_f1"] for item in summary_items]
    pixel_accuracies = [item["pixel_accuracy"] for item in summary_items]
    mean_ious = [item["mean_iou_non_background"] for item in summary_items]

    overall_summary = {
        "num_test_images": len(summary_items),
        "labeling_accuracy_exact_match": float(sum(exact_matches) / len(exact_matches)),
        "mean_label_precision": float(sum(label_precisions) / len(label_precisions)),
        "mean_label_recall": float(sum(label_recalls) / len(label_recalls)),
        "mean_label_f1": float(sum(label_f1s) / len(label_f1s)),
        "mean_pixel_accuracy": float(sum(pixel_accuracies) / len(pixel_accuracies)),
        "mean_segmentation_iou_non_background": float(sum(mean_ious) / len(mean_ious)),
        "per_class_mean_iou": {
            class_name: float(sum(values) / len(values)) if values else 0.0
            for class_name, values in class_iou_totals.items()
        },
        "samples": summary_items,
    }

    summary_path = export_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(overall_summary, fh, indent=2)

    print(f"Saved test prediction summary to: {summary_path}")
    print(f"Saved {len(summary_items)} test overlay images to: {images_dir}")
    print(f"Saved {len(summary_items)} per-image JSON reports to: {json_dir}")
    print(
        "Overall metrics | "
        f"label_exact_match={overall_summary['labeling_accuracy_exact_match']:.4f} | "
        f"label_f1={overall_summary['mean_label_f1']:.4f} | "
        f"pixel_accuracy={overall_summary['mean_pixel_accuracy']:.4f} | "
        f"mean_iou={overall_summary['mean_segmentation_iou_non_background']:.4f}"
    )


if __name__ == "__main__":
    main()
