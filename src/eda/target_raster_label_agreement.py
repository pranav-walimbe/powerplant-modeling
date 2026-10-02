"""Compare physical target-raster direction with emissions-change labels."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from config import DATASET_DF, MODEL_CLASS_NAMES, VIS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

SPLITS = ("train", "val", "test")
DEFAULT_TRIM_FRACTION = 0.05
THRESHOLD_QUANTILES = np.linspace(0.0, 0.99, 200)


def parse_args() -> argparse.Namespace:
    """Parse target-raster agreement analysis options.

    Returns:
        Target paths, metadata location, and output settings.
    """
    job_id = os.getenv("SLURM_JOB_ID", "local")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--dataframe-dir", type=Path, default=Path(DATASET_DF))
    parser.add_argument("--trim-fraction", type=float, default=DEFAULT_TRIM_FRACTION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(VIS_DIR) / f"target-raster-label-agreement-{job_id}.png",
    )
    return parser.parse_args()


def _load_frames(
    target_dir: Path,
    dataframe_dir: Path,
    trim_fraction: float,
) -> dict[str, pd.DataFrame]:
    # Calculate one signed raster score per record
    class_indices = {name: index for index, name in enumerate(MODEL_CLASS_NAMES)}
    frames = {}
    for split in SPLITS:
        targets = np.load(target_dir / f"{split}_delta_targets.npy", mmap_mode="r")
        ordered = np.sort(np.asarray(targets).reshape(len(targets), -1), axis=1)
        trim_count = int(ordered.shape[1] * trim_fraction)
        scores = ordered[:, trim_count:-trim_count].mean(axis=1)
        source = pd.read_csv(dataframe_dir / f"{split}_df.csv")
        frames[split] = pd.DataFrame(
            {
                "raster_score": scores,
                "emissions_change": source["avg_nox_end"] - source["avg_nox_start"],
                "label": source["delta_category"].map(class_indices).to_numpy(dtype=np.int64),
            }
        )
    return frames


def _implied_classes(scores: pd.Series, threshold: float) -> np.ndarray:
    # Convert a signed raster score into decrease, steady, or increase
    return np.where(scores <= -threshold, 0, np.where(scores >= threshold, 2, 1))


def _select_threshold(train: pd.DataFrame) -> tuple[np.ndarray, list[float], int]:
    # Maximize training macro F1 over symmetric deadbands
    thresholds = np.unique(np.quantile(np.abs(train["raster_score"]), THRESHOLD_QUANTILES))
    scores = [
        f1_score(
            train["label"],
            _implied_classes(train["raster_score"], threshold),
            average="macro",
            zero_division=0,
        )
        for threshold in thresholds
    ]
    return thresholds, scores, int(np.argmax(scores))


def _plot_continuous_relationship(axis: plt.Axes, frame: pd.DataFrame) -> None:
    # Plot record density against continuous emissions change
    axis.hexbin(
        frame["emissions_change"],
        frame["raster_score"],
        gridsize=60,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    pearson = frame[["emissions_change", "raster_score"]].corr().iloc[0, 1]
    spearman = frame[["emissions_change", "raster_score"]].corr(method="spearman").iloc[0, 1]
    axis.axhline(0.0, color="black", linewidth=1)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set(
        title=f"Continuous changes (Pearson {pearson:.3f}, Spearman {spearman:.3f})",
        xlabel="Emissions window change (lb/hr)",
        ylabel="Target-raster trimmed mean",
    )


def _plot_score_distributions(axis: plt.Axes, frame: pd.DataFrame, threshold: float) -> None:
    # Compare raster scores among emissions classes
    distributions = [
        frame.loc[frame["label"] == index, "raster_score"].to_numpy()
        for index in range(len(MODEL_CLASS_NAMES))
    ]
    axis.boxplot(distributions, tick_labels=[name.title() for name in MODEL_CLASS_NAMES], showfliers=False)
    axis.axhline(-threshold, color="#3977af", linestyle="--", linewidth=1.5)
    axis.axhline(threshold, color="#d65f4a", linestyle="--", linewidth=1.5)
    axis.set(
        title="Raster score by emissions label",
        xlabel="Emissions label",
        ylabel="Target-raster trimmed mean",
    )


def _plot_confusion(axis: plt.Axes, frame: pd.DataFrame, split: str, threshold: float) -> None:
    # Plot row-normalized agreement for one split
    implied = _implied_classes(frame["raster_score"], threshold)
    matrix = confusion_matrix(frame["label"], implied, labels=np.arange(len(MODEL_CLASS_NAMES)))
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    macro_f1 = f1_score(frame["label"], implied, average="macro", zero_division=0)
    balanced_accuracy = balanced_accuracy_score(frame["label"], implied)
    class_labels = [name.title() for name in MODEL_CLASS_NAMES]
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar=split == "test",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=axis,
    )
    axis.set(
        title=f"{split.title()}: macro F1 {macro_f1:.3f}, balanced acc. {balanced_accuracy:.3f}",
        xlabel="Raster-implied class",
        ylabel="Emissions label",
    )


def plot_target_label_agreement(
    target_dir: Path,
    dataframe_dir: Path,
    output_path: Path,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
) -> None:
    """Create the target-raster and emissions-label agreement figure.

    Args:
        target_dir: Directory containing physical split target arrays.
        dataframe_dir: Directory containing split metadata CSVs.
        output_path: Destination PNG path.
        trim_fraction: Fraction removed from each raster tail before averaging.
    """
    frames = _load_frames(target_dir, dataframe_dir, trim_fraction)
    thresholds, threshold_scores, selected_index = _select_threshold(frames["train"])
    selected_threshold = float(thresholds[selected_index])
    combined = pd.concat(frames.values(), ignore_index=True)

    sns.set_theme(style="whitegrid", font_scale=0.9)
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    _plot_continuous_relationship(axes[0, 0], combined)
    _plot_score_distributions(axes[0, 1], combined, selected_threshold)
    axes[0, 2].plot(thresholds, threshold_scores, color="#4C78A8", linewidth=2)
    axes[0, 2].scatter(
        [selected_threshold],
        [threshold_scores[selected_index]],
        color="#d65f4a",
        zorder=3,
    )
    axes[0, 2].set(
        title=f"Training threshold selection (tau={selected_threshold:.4g})",
        xlabel="Symmetric raster-score threshold",
        ylabel="Training macro F1",
    )
    for axis, split in zip(axes[1], SPLITS, strict=True):
        _plot_confusion(axis, frames[split], split, selected_threshold)
    figure.suptitle("Target-raster and emissions-label agreement", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run the target-raster agreement analysis."""
    args = parse_args()
    plot_target_label_agreement(
        args.target_dir,
        args.dataframe_dir,
        args.output,
        args.trim_fraction,
    )


if __name__ == "__main__":
    main()
