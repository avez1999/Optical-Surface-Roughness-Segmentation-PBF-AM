#!/usr/bin/env python3
"""Create journal-style loss-history figures from completed U-Net retraining runs."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot U-Net training and validation loss histories.")
    parser.add_argument("--polished-run", type=Path, required=True)
    parser.add_argument("--asbuilt-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


COLORS = {
    "polished": "#0072B2",
    "asbuilt": "#D55E00",
    "validation": "#009E73",
    "best": "#3A3A3A",
}


def load_history(run_dir: Path) -> list[dict]:
    path = run_dir / "training_history.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot(run_dir: Path, output_name: str, color: str, output_dir: Path) -> None:
    history = load_history(run_dir)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    validation_loss = [row["val_loss"] for row in history]
    best_epoch = min(history, key=lambda row: row["val_loss"])["epoch"]
    best_loss = min(validation_loss)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(3.54, 2.7), constrained_layout=True)
    axis.plot(epochs, train_loss, color=color, linewidth=1.35, label="Training")
    axis.plot(epochs, validation_loss, color=COLORS["validation"], linewidth=1.35, label="Validation")
    axis.scatter(best_epoch, best_loss, s=18, color=COLORS["validation"], edgecolor="white", linewidth=0.55, zorder=4)
    axis.axvline(best_epoch, color=COLORS["best"], linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_xlim(1, max(epochs))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.75, color="#333333")
    axis.legend(loc="upper right", frameon=False, handlelength=2.0)
    axis.text(
        0.02,
        0.97,
        f"Best validation epoch: {best_epoch}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color=COLORS["best"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / output_name
    figure.savefig(png_path, dpi=600, facecolor="white")
    figure.savefig(png_path.with_suffix(".tif"), dpi=600, facecolor="white")
    plt.close(figure)
    print(f"Saved {png_path} from {len(history)} epochs")


def main() -> None:
    args = parse_args()
    plot(
        args.polished_run,
        "polished_vanilla_unet_loss_history.png",
        COLORS["polished"],
        args.output_dir,
    )
    plot(
        args.asbuilt_run,
        "asbuilt_vanilla_unet_loss_history.png",
        COLORS["asbuilt"],
        args.output_dir,
    )


if __name__ == "__main__":
    main()
