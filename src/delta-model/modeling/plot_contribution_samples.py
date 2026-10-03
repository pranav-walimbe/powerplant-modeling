"""Plot a target-only contact sheet of decrease and increase contributions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from modeling.dataset import (
    CONTRIBUTION_CHANNEL_NAMES,
    SOURCE_PIXEL_NOX_COLUMNS,
    TIMESTEP_TIME_COLUMNS,
    build_contribution_target,
)

from config import DATASET_DF, MODEL_TARGET_COL

PLOTTED_CLASSES = ("decrease", "increase")
DEFAULT_SAMPLES_PER_CLASS = 100
DEFAULT_COLUMNS = 20


def parse_args() -> argparse.Namespace:
    """Parse contribution contact-sheet options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--dataframe-dir", default=DATASET_DF)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _evenly_spaced_positions(positions: np.ndarray, count: int) -> np.ndarray:
    # Select deterministic positions spanning the ordered candidate range
    if count >= len(positions):
        return positions
    ranks = np.rint(np.linspace(0, len(positions) - 1, count)).astype(np.int64)
    return positions[ranks]


def _select_diverse_indices(frame: pd.DataFrame, class_name: str, count: int) -> np.ndarray:
    # Cover every available AOI before filling from the class-wide change spectrum
    candidates = frame.index[frame[MODEL_TARGET_COL] == class_name].to_numpy(dtype=np.int64)
    if len(candidates) < count:
        raise ValueError(f"Requested {count} {class_name} samples but found {len(candidates)}")

    representative_indices = []
    for _, group in frame.loc[candidates].groupby("aoi_id", sort=True):
        group_indices = group.index.to_numpy(dtype=np.int64)
        group_changes = frame.loc[group_indices, "nox_change"].to_numpy(dtype=np.float64)
        median_change = np.median(group_changes)
        representative_indices.append(group_indices[np.argmin(np.abs(group_changes - median_change))])

    representatives = np.asarray(representative_indices, dtype=np.int64)
    if len(representatives) > count:
        order = np.lexsort((representatives, frame.loc[representatives, "nox_change"]))
        representatives = _evenly_spaced_positions(representatives[order], count)

    remaining_count = count - len(representatives)
    if remaining_count:
        remaining = np.setdiff1d(candidates, representatives, assume_unique=False)
        order = np.lexsort((remaining, frame.loc[remaining, "nox_change"]))
        representatives = np.concatenate(
            (representatives, _evenly_spaced_positions(remaining[order], remaining_count))
        )

    order = np.lexsort((representatives, frame.loc[representatives, "nox_change"]))
    return representatives[order]


def _elapsed_hours(frame: pd.DataFrame) -> np.ndarray:
    # Convert the five UTC observation times into four transition durations
    timestamps = np.column_stack(
        [
            pd.to_datetime(frame[column], utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
            for column in TIMESTEP_TIME_COLUMNS
        ]
    )
    return (np.diff(timestamps, axis=1) / np.timedelta64(1, "h")).astype(np.float32)


def _build_targets(
    frame: pd.DataFrame,
    elapsed_hours: np.ndarray,
    selected_indices: dict[str, np.ndarray],
    sigma: float,
) -> dict[str, np.ndarray]:
    # Build only the active class channel for the selected records
    targets = {}
    for class_name, indices in selected_indices.items():
        channel = CONTRIBUTION_CHANNEL_NAMES.index(class_name)
        class_targets = []
        for index in indices:
            row = frame.loc[index]
            target = build_contribution_target(
                row.loc[list(SOURCE_PIXEL_NOX_COLUMNS)].to_numpy(dtype=object),
                elapsed_hours[index],
                class_name,
                sigma=sigma,
            )
            class_targets.append(target[channel])
        targets[class_name] = np.stack(class_targets)
    return targets


def plot_contribution_samples(
    targets: dict[str, np.ndarray],
    output: Path,
    *,
    split: str,
    sigma: float,
    columns: int,
) -> None:
    """Save decrease and increase contribution targets as one contact sheet.

    Args:
        targets: Active contribution rasters keyed by class name.
        output: Destination PNG path.
        split: Dataset split represented by the samples.
        sigma: Gaussian splat width used to construct the targets.
        columns: Number of rasters per contact-sheet row.
    """
    sample_count = len(next(iter(targets.values())))
    rows_per_class = int(np.ceil(sample_count / columns))
    total_rows = rows_per_class * len(PLOTTED_CLASSES)
    figure, axes = plt.subplots(
        total_rows,
        columns,
        figsize=(columns * 1.15, total_rows * 1.15),
        squeeze=False,
    )
    maximum = max(float(values.max()) for values in targets.values())
    normalization = PowerNorm(gamma=0.5, vmin=0.0, vmax=maximum)
    image = None
    for class_offset, class_name in enumerate(PLOTTED_CLASSES):
        row_offset = class_offset * rows_per_class
        for sample_index, target in enumerate(targets[class_name]):
            row = row_offset + sample_index // columns
            column = sample_index % columns
            axis = axes[row, column]
            image = axis.imshow(target, cmap="magma", norm=normalization, interpolation="nearest")
            axis.set_xticks(())
            axis.set_yticks(())
        for unused_index in range(sample_count, rows_per_class * columns):
            row = row_offset + unused_index // columns
            column = unused_index % columns
            axes[row, column].set_visible(False)

    figure.text(0.006, 0.72, "Decrease", rotation=90, va="center", fontsize=14, weight="bold")
    figure.text(0.006, 0.28, "Increase", rotation=90, va="center", fontsize=14, weight="bold")
    split_name = "Validation" if split == "val" else split.title()
    figure.suptitle(
        f"{split_name} contribution targets: {sample_count} decrease + "
        f"{sample_count} increase (splat sigma {sigma:g})",
        fontsize=16,
    )
    figure.subplots_adjust(left=0.035, right=0.94, top=0.93, bottom=0.07, wspace=0.04, hspace=0.04)
    colorbar_axis = figure.add_axes((0.20, 0.025, 0.60, 0.018))
    figure.colorbar(image, cax=colorbar_axis, orientation="horizontal").set_label(
        "Normalized facility-change contribution (square-root color scale)"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Build and save a diverse contribution-target contact sheet."""
    args = parse_args()
    columns = (
        "aoi_id",
        "avg_nox_start",
        "avg_nox_end",
        MODEL_TARGET_COL,
        *TIMESTEP_TIME_COLUMNS,
        *SOURCE_PIXEL_NOX_COLUMNS,
    )
    dataframe_path = Path(args.dataframe_dir) / f"{args.split}_df.csv"
    frame = pd.read_csv(dataframe_path, usecols=columns)
    frame["nox_change"] = frame["avg_nox_end"] - frame["avg_nox_start"]
    selected_indices = {
        class_name: _select_diverse_indices(frame, class_name, args.samples_per_class)
        for class_name in PLOTTED_CLASSES
    }
    targets = _build_targets(frame, _elapsed_hours(frame), selected_indices, args.sigma)
    plot_contribution_samples(
        targets,
        args.output,
        split=args.split,
        sigma=args.sigma,
        columns=args.columns,
    )
    print(
        f"Saved {args.samples_per_class} decrease and {args.samples_per_class} increase "
        f"contribution rasters to {args.output}"
    )


if __name__ == "__main__":
    main()
