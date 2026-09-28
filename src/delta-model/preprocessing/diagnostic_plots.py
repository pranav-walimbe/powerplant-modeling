"""Shared diagnostic plots for delta-model preprocessing."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes

from config import (
    SEQUENCE_TIMESTEPS,
    STRATIFICATION_WINDOW_ABSOLUTE_FLOOR,
    STRATIFICATION_WINDOW_RELATIVE_FLOOR,
)

AOI_ID_COL = "aoi_id"
DELTA_CATEGORY_COL = "delta_category"
CLASS_NAMES = ("decrease", "steady", "increase")
CLASS_COLORS = ("#3977af", "#999999", "#d65f4a")


def plot_window_emissions_axes(splits: Mapping[str, pl.DataFrame], axes: Sequence[Axes]) -> None:
    """Plot class-specific timestep emissions on one axis per split.

    Args:
        splits: Ordered split names mapped to records carrying timestep NOx and labels.
        axes: One Matplotlib axis per split.
    """
    if len(splits) != len(axes):
        raise ValueError(f"Expected {len(splits)} axes but received {len(axes)}")
    positions = np.arange(SEQUENCE_TIMESTEPS)
    for column_index, ((split_name, frame), axis) in enumerate(zip(splits.items(), axes, strict=True)):
        for class_name, color in zip(CLASS_NAMES, CLASS_COLORS, strict=True):
            class_frame = frame.filter(pl.col(DELTA_CATEGORY_COL) == class_name)
            if class_frame.is_empty():
                continue
            values = np.column_stack(
                [class_frame[f"t{index}_nox"].to_numpy() for index in range(SEQUENCE_TIMESTEPS)]
            )
            median = np.median(values, axis=0)
            lower, upper = np.quantile(values, (0.25, 0.75), axis=0)
            axis.plot(positions, median, marker="o", color=color, label=class_name.title())
            axis.fill_between(positions, lower, upper, color=color, alpha=0.15)
        axis.set(
            title=f"{split_name.title()}: {frame.height:,} records, {frame[AOI_ID_COL].n_unique():,} AOIs",
            xlabel="TEMPO timestep",
            ylabel="Interpolated NOx (lb/hr)",
            xticks=positions,
            xticklabels=[f"t{index}" for index in positions],
        )
        axis.grid(alpha=0.2)
        if column_index == 0:
            axis.legend()


def plot_post_filter_window_behavior(splits: Mapping[str, pl.DataFrame], output_path: Path) -> None:
    """Save emissions trajectories for records that survived raster filtering.

    Args:
        splits: Ordered split names mapped to post-filter records.
        output_path: Destination PNG path.
    """
    figure, raw_axes = plt.subplots(1, len(splits), figsize=(16, 4.8), constrained_layout=True)
    axes = [raw_axes] if len(splits) == 1 else list(raw_axes)
    plot_window_emissions_axes(splits, axes)
    figure.suptitle(
        "Five-timestep emissions behavior after raster filtering\n"
        f"Threshold: max({STRATIFICATION_WINDOW_ABSOLUTE_FLOOR:g} lb/hr, "
        f"{STRATIFICATION_WINDOW_RELATIVE_FLOOR:.0%} of early-late midpoint); lines show median and IQR"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
