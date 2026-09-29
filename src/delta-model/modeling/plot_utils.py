"""Plotting utilities for delta-model classifier comparisons."""

import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from modeling.dataset import FINAL_TIMESTEP_TIME_COLUMN, midpoint_local_solar_hour
from modeling.eval_utils import PREDICTED_CLASS_COL, TRUE_CLASS_COL, classification_metrics

from config import MODEL_CLASS_NAMES

MODEL_ORDER = ("seasonal", "transport_image_only", "transport_context")
MODEL_DISPLAY_NAMES = {
    "seasonal": "Seasonal",
    "transport_image_only": "Transport image-only",
    "transport_context": "Transport + context",
}
MODEL_COLORS = {
    "seasonal": "#4C78A8",
    "transport_image_only": "#54A24B",
    "transport_context": "#B279A2",
}
SPLIT_ORDER = ("train", "val", "test")
STRATUM_ORDER = ("Low", "Middle", "High")
AOI_STRATA = {
    "avg_heat_input": "Average heat input",
    "num_units": "Total unit count",
}
MONTH_CHARACTERISTIC = "month"
SOLAR_HOUR_CHARACTERISTIC = "midpoint_local_solar_hour"
CHARACTERISTIC_PANELS = (
    ("avg_heat_input", "test", "Average heat input"),
    ("num_units", "test", "Total unit count"),
    (MONTH_CHARACTERISTIC, "test", "Month (test)"),
    (SOLAR_HOUR_CHARACTERISTIC, "test", "Midpoint local solar hour (test)"),
)


def _save(figure: plt.Figure, run_dir: str | Path, plot_name: str) -> None:
    # Persist and close one completed plot
    figure.savefig(Path(run_dir) / f"{plot_name}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _available_models(model_frames: dict[str, dict[str, pd.DataFrame]]) -> tuple[str, ...]:
    # Keep model order consistent across artifacts
    return tuple(name for name in MODEL_ORDER if name in model_frames)


def _label_bars(axis: Axes, bars: BarContainer) -> None:
    # Display each plotted accuracy above its bar
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)


def plot_split_class_accuracy(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    run_dir: str | Path,
) -> None:
    """Plot overall and per-class accuracy for each data split.

    Args:
        model_frames: Row-level predictions by model and split.
        run_dir: Model-run output directory.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    models = _available_models(model_frames)
    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 1.25))
    overall_axis = figure.add_subplot(grid[0, :])
    class_axes = [figure.add_subplot(grid[1, index]) for index in range(len(SPLIT_ORDER))]

    split_positions = np.arange(len(SPLIT_ORDER), dtype=np.float64)
    width = 0.8 / len(models)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
    for model_name, offset in zip(models, offsets, strict=True):
        scores = [classification_metrics(model_frames[model_name][split])["accuracy"] for split in SPLIT_ORDER]
        bars = overall_axis.bar(
            split_positions + offset,
            scores,
            width=width,
            color=MODEL_COLORS[model_name],
            label=MODEL_DISPLAY_NAMES[model_name],
        )
        _label_bars(overall_axis, bars)
    overall_axis.set(
        title="Overall accuracy by split",
        ylabel="Accuracy",
        xticks=split_positions,
        xticklabels=[split.title() for split in SPLIT_ORDER],
        ylim=(0, 1.08),
    )
    overall_axis.legend(loc="lower right")

    class_positions = np.arange(len(MODEL_CLASS_NAMES), dtype=np.float64)
    for axis, split in zip(class_axes, SPLIT_ORDER, strict=True):
        for model_name, offset in zip(models, offsets, strict=True):
            recalls = classification_metrics(model_frames[model_name][split])["class_recall"]
            scores = [recalls[class_name] for class_name in MODEL_CLASS_NAMES]
            bars = axis.bar(
                class_positions + offset,
                scores,
                width=width,
                color=MODEL_COLORS[model_name],
                label=MODEL_DISPLAY_NAMES[model_name],
            )
            _label_bars(axis, bars)
        axis.set(
            title=f"{split.title()} per-class accuracy",
            ylabel="Recall",
            xticks=class_positions,
            xticklabels=[name.title() for name in MODEL_CLASS_NAMES],
            ylim=(0, 1.08),
        )
    figure.suptitle("Classification accuracy by model", fontsize=16)
    figure.tight_layout()
    _save(figure, run_dir, "split_class_accuracy")


def plot_training_curves(
    histories: dict[str, tuple[list[float], list[float]]],
    run_dir: str | Path,
) -> None:
    """Plot train and validation loss for each classifier.

    Args:
        histories: Train and validation losses by model.
        run_dir: Model-run output directory.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    models = tuple(name for name in MODEL_ORDER if name in histories)
    columns = min(2, len(models))
    rows = int(np.ceil(len(models) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(7.5 * columns, 5.5 * rows), squeeze=False)
    flat_axes = axes.ravel()
    for axis, model_name in zip(flat_axes, models, strict=False):
        train_losses, validation_losses = histories[model_name]
        axis.plot(range(1, len(train_losses) + 1), train_losses, label="Train", linewidth=2)
        axis.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation", linewidth=2)
        axis.set(
            title=MODEL_DISPLAY_NAMES[model_name],
            xlabel="Epoch",
            ylabel="Cross-entropy",
        )
        axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axis.legend()
    for axis in flat_axes[len(models) :]:
        axis.set_visible(False)
    figure.suptitle("Training and validation loss", fontsize=16)
    figure.tight_layout()
    _save(figure, run_dir, "training_curves")


def _accuracy(frame: pd.DataFrame) -> float:
    # Calculate row-level classification accuracy
    return float((frame[TRUE_CLASS_COL] == frame[PREDICTED_CLASS_COL]).mean())


def _tertiles(values: pd.Series) -> pd.Series:
    # Assign equally ranked low, middle, and high groups
    percentiles = values.rank(method="average", pct=True)
    return pd.cut(
        percentiles,
        bins=(0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
        labels=STRATUM_ORDER,
        include_lowest=True,
    )


def _stratum_result(
    frames: dict[str, pd.DataFrame],
    *,
    characteristic: str,
    characteristic_label: str,
    split: str,
    stratum: str,
    mask: pd.Series,
    values: pd.Series,
    stratification_unit: str,
) -> dict[str, object]:
    # Summarize one model comparison group
    reference = next(iter(frames.values()))
    row = {
        "characteristic": characteristic,
        "characteristic_label": characteristic_label,
        "split": split,
        "stratification_unit": stratification_unit,
        "stratum": stratum,
        "aoi_count": int(reference.loc[mask, "aoi_id"].nunique()),
        "record_count": int(mask.sum()),
        "value_min": float(values.min()),
        "value_max": float(values.max()),
    }
    row.update({f"{model_name}_accuracy": _accuracy(frame.loc[mask]) for model_name, frame in frames.items()})
    return row


def _aligned_frames(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    split: str,
) -> dict[str, pd.DataFrame]:
    # Return row-aligned predictions for one split
    frames = {
        model_name: model_frames[model_name][split].reset_index(drop=True)
        for model_name in _available_models(model_frames)
    }
    reference_name = next(iter(frames))
    reference = frames[reference_name][["aoi_id", TRUE_CLASS_COL]]
    for model_name, frame in frames.items():
        if not reference.equals(frame[["aoi_id", TRUE_CLASS_COL]]):
            raise ValueError(f"{reference_name} and {model_name} {split} predictions are not row-aligned")
    return frames


def _characteristic_accuracy_table(model_frames: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    # Compare model accuracy across AOI properties, month, and midpoint solar hour
    test_frames = _aligned_frames(model_frames, "test")
    test_reference = next(iter(test_frames.values()))

    aoi_values = test_reference.groupby("aoi_id", sort=True)[list(AOI_STRATA)].median(numeric_only=True)
    rows: list[dict[str, object]] = []
    for column, display_name in AOI_STRATA.items():
        strata = _tertiles(aoi_values[column])
        for stratum in STRATUM_ORDER:
            aoi_ids = strata.index[strata == stratum]
            mask = test_reference["aoi_id"].isin(aoi_ids)
            if not mask.any():
                continue
            rows.append(
                _stratum_result(
                    test_frames,
                    characteristic=column,
                    characteristic_label=display_name,
                    split="test",
                    stratum=stratum,
                    mask=mask,
                    values=aoi_values.loc[aoi_ids, column],
                    stratification_unit="AOI",
                )
            )

    months = pd.to_datetime(test_reference[FINAL_TIMESTEP_TIME_COLUMN], utc=True).dt.month
    for month in sorted(months.unique()):
        mask = months == month
        rows.append(
            _stratum_result(
                test_frames,
                characteristic=MONTH_CHARACTERISTIC,
                characteristic_label="Month (test)",
                split="test",
                stratum=calendar.month_abbr[month],
                mask=mask,
                values=months.loc[mask],
                stratification_unit="record",
            )
        )

    solar_hours = midpoint_local_solar_hour(test_reference)
    solar_hour_bins = np.floor(solar_hours).astype(int)
    for hour in sorted(solar_hour_bins.unique()):
        mask = solar_hour_bins == hour
        rows.append(
            _stratum_result(
                test_frames,
                characteristic=SOLAR_HOUR_CHARACTERISTIC,
                characteristic_label="Midpoint local solar hour (test)",
                split="test",
                stratum=f"{hour:02d}:00",
                mask=mask,
                values=solar_hours.loc[mask],
                stratification_unit="record",
            )
        )
    return pd.DataFrame(rows)


def plot_accuracy_by_characteristic(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    run_dir: str | Path,
) -> None:
    """Plot accuracy by test AOI characteristics, month, and midpoint solar hour.

    Args:
        model_frames: Row-level predictions by model and split.
        run_dir: Model-run output directory.
    """
    table = _characteristic_accuracy_table(model_frames)
    table.to_csv(Path(run_dir) / "accuracy_by_characteristic.csv", index=False)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    columns = 2
    rows = int(np.ceil(len(CHARACTERISTIC_PANELS) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 5 * rows), sharey=True, squeeze=False)
    flat_axes = axes.ravel()
    models = _available_models(model_frames)
    for axis, (column, split, display_name) in zip(flat_axes, CHARACTERISTIC_PANELS, strict=True):
        subset = table.loc[(table["characteristic"] == column) & (table["split"] == split)]
        if column in AOI_STRATA:
            subset = subset.set_index("stratum").reindex(STRATUM_ORDER).reset_index()
        else:
            subset = subset.sort_values("value_min")
        positions = np.arange(len(subset), dtype=np.float64)
        for model_name in models:
            axis.plot(
                positions,
                subset[f"{model_name}_accuracy"].to_numpy(dtype=np.float64),
                marker="o",
                linewidth=2,
                color=MODEL_COLORS[model_name],
                label=MODEL_DISPLAY_NAMES[model_name],
            )
        if column in AOI_STRATA:
            tick_labels = [
                f"{row['stratum']}\n{row['value_min']:.3g}-{row['value_max']:.3g}"
                for _, row in subset.iterrows()
            ]
            x_label = "AOI tertile"
        elif column == MONTH_CHARACTERISTIC:
            tick_labels = subset["stratum"].tolist()
            x_label = "Month"
        else:
            tick_labels = subset["stratum"].tolist()
            x_label = "Midpoint local solar hour"
        axis.set(
            title=display_name,
            xlabel=x_label,
            ylabel=f"{split.title()} accuracy",
            xticks=positions,
            xticklabels=tick_labels,
            ylim=(0, 1.05),
        )
    flat_axes[0].legend(loc="lower right", fontsize=8)
    figure.suptitle(
        "Model accuracy by AOI characteristic, calendar month, and midpoint solar hour",
        fontsize=16,
    )
    figure.tight_layout()
    _save(figure, run_dir, "accuracy_by_characteristic")
