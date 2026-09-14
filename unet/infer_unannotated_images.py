#!/usr/bin/env python3
"""Create automatic semantic-segmentation labels for images without annotations."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import torch

from train_unet import UNet


COLOR_PALETTE = {
    0: (0, 0, 0),
    1: (230, 57, 70),
    2: (29, 161, 242),
    3: (46, 204, 113),
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create U-Net pseudo-labels for unannotated images.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-name",
        action="append",
        default=[],
        help="Process one named image; repeat this option to process a batch of named images.",
    )
    return parser.parse_args()


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in COLOR_PALETTE.items():
        color_mask[mask == class_id] = color
    return Image.fromarray(color_mask, mode="RGB")


def main() -> None:
    args = parse_args()
    image_paths = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.image_name:
        requested_names = set(args.image_name)
        image_paths = [path for path in image_paths if path.name in requested_names]
        found_names = {path.name for path in image_paths}
        missing_names = requested_names - found_names
        if missing_names:
            raise ValueError(f"Requested image names were not found: {sorted(missing_names)}")
    if not image_paths:
        raise ValueError(f"No supported image files found in {args.input_dir}.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_map: Dict[str, int] = checkpoint["class_map"]
    image_size = checkpoint["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        in_channels=3,
        num_classes=len(class_map),
        base_channels=checkpoint["base_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    optical_dir = args.output_dir / "optical"
    overlay_dir = args.output_dir / "prediction_overlay"
    mask_dir = args.output_dir / "prediction_mask"
    for directory in (optical_dir, overlay_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    class_map_inv = {class_id: label for label, class_id in class_map.items()}
    summary_items: List[Dict[str, object]] = []
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            model_image = image
            if image_size is not None:
                model_image = image.resize((image_size, image_size), Image.BILINEAR)
            logits = model(image_to_tensor(model_image).unsqueeze(0).to(device))
            mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            if image_size is not None:
                mask = np.asarray(
                    Image.fromarray(mask).resize(original_size, Image.NEAREST), dtype=np.uint8
                )

            color_mask = colorize_mask(mask)
            overlay = Image.blend(image, color_mask, alpha=0.4)
            stem = image_path.stem
            image.save(optical_dir / f"{stem}.png")
            overlay.save(overlay_dir / f"{stem}_prediction_overlay.png")
            Image.fromarray(mask, mode="L").save(mask_dir / f"{stem}_class_index_mask.png")

            total_pixels = int(mask.size)
            pixel_counts = {
                class_map_inv[class_id]: int((mask == class_id).sum())
                for class_id in sorted(class_map_inv)
            }
            pixel_percentages = {
                class_name: 100.0 * count / total_pixels for class_name, count in pixel_counts.items()
            }
            summary_items.append(
                {
                    "image_name": image_path.name,
                    "source_size_px": {"width": original_size[0], "height": original_size[1]},
                    "predicted_pixel_counts": pixel_counts,
                    "predicted_pixel_percentages": pixel_percentages,
                }
            )

    summary_path = args.output_dir / "summary.json"
    existing_items = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            existing_items = json.load(handle).get("images", [])
    items_by_name = {item["image_name"]: item for item in existing_items}
    items_by_name.update({item["image_name"]: item for item in summary_items})

    summary = {
        "checkpoint": str(args.checkpoint),
        "source_directory": str(args.input_dir),
        "dataset_role": "unannotated inference only; no ground truth is available for metric calculation",
        "class_map": class_map,
        "overlay_palette_rgb": {class_map_inv[key]: value for key, value in COLOR_PALETTE.items() if key in class_map_inv},
        "images": [items_by_name[name] for name in sorted(items_by_name)],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved automatic annotations for {len(summary_items)} images to: {args.output_dir}")


if __name__ == "__main__":
    main()
