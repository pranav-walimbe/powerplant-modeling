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
from sklearn.metrics import f1_score

from config import MODEL_CLASS_NAMES

MODEL_ORDER = ("seasonal", "vision")
MODEL_DISPLAY_NAMES = {
    "seasonal": "Seasonal",
    "vision": "Vision",
}
MODEL_COLORS = {
    "seasonal": "#4C78A8",
    "vision": "#B279A2",
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
    ("avg_heat_input", "test", "Average heat input", "macro_f1"),
    ("num_units", "test", "Total unit count", "macro_f1"),
    (MONTH_CHARACTERISTIC, "test", "Month (test)", "accuracy"),
    (SOLAR_HOUR_CHARACTERISTIC, "test", "Midpoint local solar hour (test)", "accuracy"),
)


def _save(figure: plt.Figure, run_dir: str | Path, plot_name: str) -> None:
    # Persist and close one completed plot
    figure.savefig(Path(run_dir) / f"{plot_name}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _available_models(model_frames: dict[str, dict[str, pd.DataFrame]]) -> tuple[str, ...]:
    # Keep model order consistent across artifacts
    return tuple(name for name in MODEL_ORDER if name in model_frames)


def _label_bars(axis: Axes, bars: BarContainer) -> None:
    # Display each plotted score above its bar
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)


def plot_split_class_f1(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    run_dir: str | Path,
) -> None:
    """Plot macro and per-class F1 scores for each data split.

    Args:
        model_frames: Row-level predictions by model and split.
        run_dir: Model-run output directory.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    models = _available_models(model_frames)
    metrics = {
        model_name: {
            split: classification_metrics(model_frames[model_name][split]) for split in SPLIT_ORDER
        }
        for model_name in models
    }
    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(2, len(SPLIT_ORDER), height_ratios=(1.0, 1.25))
    macro_axis = figure.add_subplot(grid[0, :])
    class_axes = [figure.add_subplot(grid[1, index]) for index in range(len(SPLIT_ORDER))]
    width = 0.8 / len(models)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width

    split_positions = np.arange(len(SPLIT_ORDER), dtype=np.float64)
    for model_name, offset in zip(models, offsets, strict=True):
        scores = [metrics[model_name][split]["macro_f1"] for split in SPLIT_ORDER]
        bars = macro_axis.bar(
            split_positions + offset,
            scores,
            width=width,
            color=MODEL_COLORS[model_name],
            label=MODEL_DISPLAY_NAMES[model_name],
        )
        _label_bars(macro_axis, bars)
    macro_axis.set(
        title="Macro F1 by split",
        ylabel="Macro F1",
        xticks=split_positions,
        xticklabels=[split.title() for split in SPLIT_ORDER],
        ylim=(0, 1.08),
    )
    macro_axis.legend(loc="lower right")

    class_positions = np.arange(len(MODEL_CLASS_NAMES), dtype=np.float64)
    for axis, split in zip(class_axes, SPLIT_ORDER, strict=True):
        class_counts = metrics[models[0]][split]["class_counts"]
        for model_name, offset in zip(models, offsets, strict=True):
            class_f1 = metrics[model_name][split]["class_f1"]
            scores = [class_f1[class_name] for class_name in MODEL_CLASS_NAMES]
            bars = axis.bar(
                class_positions + offset,
                scores,
                width=width,
                color=MODEL_COLORS[model_name],
                label=MODEL_DISPLAY_NAMES[model_name],
            )
            _label_bars(axis, bars)
        axis.set(
            title=f"{split.title()} per-class F1",
            ylabel="F1 score",
            xticks=class_positions,
            xticklabels=[f"{name.title()}\n(n={class_counts[name]:,})" for name in MODEL_CLASS_NAMES],
            ylim=(0, 1.08),
        )
    figure.suptitle("Macro and per-class F1 by split and model", fontsize=16)
    figure.tight_layout()
    _save(figure, run_dir, "split_class_f1")


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
    metrics = {
        model_name: {
            split: classification_metrics(model_frames[model_name][split]) for split in SPLIT_ORDER
        }
        for model_name in models
    }
    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(2, len(SPLIT_ORDER), height_ratios=(1.0, 1.25))
    overall_axis = figure.add_subplot(grid[0, :])
    class_axes = [figure.add_subplot(grid[1, index]) for index in range(len(SPLIT_ORDER))]
    width = 0.8 / len(models)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width

    split_positions = np.arange(len(SPLIT_ORDER), dtype=np.float64)
    for model_name, offset in zip(models, offsets, strict=True):
        scores = [metrics[model_name][split]["overall_accuracy"] for split in SPLIT_ORDER]
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
        class_counts = metrics[models[0]][split]["class_counts"]
        for model_name, offset in zip(models, offsets, strict=True):
            class_accuracy = metrics[model_name][split]["class_accuracy"]
            scores = [class_accuracy[class_name] for class_name in MODEL_CLASS_NAMES]
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
            ylabel="Accuracy",
            xticks=class_positions,
            xticklabels=[f"{name.title()}\n(n={class_counts[name]:,})" for name in MODEL_CLASS_NAMES],
            ylim=(0, 1.08),
        )
    figure.suptitle("Overall and per-class accuracy by split and model", fontsize=16)
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


def _macro_f1(frame: pd.DataFrame) -> float:
    # Calculate fixed-label macro F1 for one group
    labels = np.arange(len(MODEL_CLASS_NAMES))
    return float(
        f1_score(
            frame[TRUE_CLASS_COL],
            frame[PREDICTED_CLASS_COL],
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


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
    metric: str,
) -> dict[str, object]:
    # Summarize one model comparison group
    reference = next(iter(frames.values()))
    row = {
        "characteristic": characteristic,
        "characteristic_label": characteristic_label,
        "split": split,
        "stratification_unit": stratification_unit,
        "metric": metric,
        "stratum": stratum,
        "aoi_count": int(reference.loc[mask, "aoi_id"].nunique()),
        "record_count": int(mask.sum()),
        "value_min": float(values.min()),
        "value_max": float(values.max()),
    }
    score = _macro_f1 if metric == "macro_f1" else _accuracy
    row.update({f"{model_name}_score": score(frame.loc[mask]) for model_name, frame in frames.items()})
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


def _characteristic_performance_table(model_frames: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    # Compare model performance across AOI properties, month, and midpoint solar hour
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
                    metric="macro_f1",
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
                metric="accuracy",
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
                metric="accuracy",
            )
        )
    return pd.DataFrame(rows)


def plot_performance_by_characteristic(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    run_dir: str | Path,
) -> None:
    """Plot test performance by AOI characteristics, month, and midpoint solar hour.

    Args:
        model_frames: Row-level predictions by model and split.
        run_dir: Model-run output directory.
    """
    table = _characteristic_performance_table(model_frames)
    table.to_csv(Path(run_dir) / "performance_by_characteristic.csv", index=False)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    columns = 2
    rows = int(np.ceil(len(CHARACTERISTIC_PANELS) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 5 * rows), sharey=True, squeeze=False)
    flat_axes = axes.ravel()
    models = _available_models(model_frames)
    for axis, (column, split, display_name, metric) in zip(flat_axes, CHARACTERISTIC_PANELS, strict=True):
        subset = table.loc[(table["characteristic"] == column) & (table["split"] == split)]
        if column in AOI_STRATA:
            subset = subset.set_index("stratum").reindex(STRATUM_ORDER).reset_index()
        else:
            subset = subset.sort_values("value_min")
        positions = np.arange(len(subset), dtype=np.float64)
        for model_name in models:
            axis.plot(
                positions,
                subset[f"{model_name}_score"].to_numpy(dtype=np.float64),
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
            ylabel=f"{split.title()} {'macro F1' if metric == 'macro_f1' else 'accuracy'}",
            xticks=positions,
            xticklabels=tick_labels,
            ylim=(0, 1.05),
        )
    flat_axes[0].legend(loc="lower right", fontsize=8)
    figure.suptitle(
        "Model performance by AOI characteristic, calendar month, and midpoint solar hour",
        fontsize=16,
    )
    figure.tight_layout()
    _save(figure, run_dir, "performance_by_characteristic")


def plot_joint_prediction_samples(
    no2_rasters: np.ndarray,
    transport_residuals: np.ndarray,
    target_rasters: np.ndarray,
    predicted_rasters: np.ndarray,
    true_classes: np.ndarray,
    predicted_classes: np.ndarray,
    nox_changes: np.ndarray,
    run_dir: str | Path,
) -> None:
    """Plot NO2 sequences, transport residuals, and joint predictions.

    Args:
        no2_rasters: Physical NO2 rasters shaped sample by five timesteps by height by width.
        transport_residuals: Physical residuals shaped sample by four transitions by height by width.
        target_rasters: Contribution targets shaped sample by channel by height by width.
        predicted_rasters: Predicted contribution rasters matching ``target_rasters``.
        true_classes: Integer class labels for each sample.
        predicted_classes: Integer predicted class labels for each sample.
        nox_changes: Signed late-minus-early NOx changes in pounds per hour.
        run_dir: Model-run output directory.
    """
    sns.set_theme(style="white", font_scale=0.8)
    no2_low, no2_high = np.nanpercentile(no2_rasters, (2.0, 98.0))
    residual_limit = np.nanpercentile(np.abs(transport_residuals), 98.0)
    target_limit = np.nanpercentile(np.concatenate((target_rasters, predicted_rasters)), 98.0)
    column_titles = (
        "NO2 t1",
        "NO2 t2",
        "NO2 t3",
        "NO2 t4",
        "NO2 t5",
        "Residual 1-2",
        "Residual 2-3",
        "Residual 3-4",
        "Residual 4-5",
        "Target decrease",
        "Target increase",
        "Predicted decrease",
        "Predicted increase",
    )
    figure, axes = plt.subplots(
        len(no2_rasters),
        len(column_titles),
        figsize=(26, 36),
        squeeze=False,
    )
    sample_labels = zip(true_classes, predicted_classes, nox_changes, strict=True)
    for row, (true_class, predicted_class, nox_change) in enumerate(sample_labels):
        for timestep in range(no2_rasters.shape[1]):
            no2_image = axes[row, timestep].imshow(
                no2_rasters[row, timestep],
                cmap="viridis",
                vmin=no2_low,
                vmax=no2_high,
            )
        for transition in range(transport_residuals.shape[1]):
            residual_image = axes[row, 5 + transition].imshow(
                transport_residuals[row, transition],
                cmap="RdBu_r",
                vmin=-residual_limit,
                vmax=residual_limit,
            )
        target_image = axes[row, 9].imshow(
            target_rasters[row, 0],
            cmap="magma",
            vmin=0,
            vmax=target_limit,
        )
        axes[row, 10].imshow(
            target_rasters[row, 1],
            cmap="magma",
            vmin=0,
            vmax=target_limit,
        )
        axes[row, 11].imshow(
            predicted_rasters[row, 0],
            cmap="magma",
            vmin=0,
            vmax=target_limit,
        )
        axes[row, 12].imshow(
            predicted_rasters[row, 1],
            cmap="magma",
            vmin=0,
            vmax=target_limit,
        )
        axes[row, 0].set_ylabel(
            f"NOx change: {nox_change:+.1f} lb/hr\n"
            f"True: {MODEL_CLASS_NAMES[true_class]}\nPred: {MODEL_CLASS_NAMES[predicted_class]}",
            rotation=0,
            horizontalalignment="right",
            verticalalignment="center",
            labelpad=32,
            fontsize=7,
        )
        for axis in axes[row]:
            axis.set_xticks(())
            axis.set_yticks(())
    for axis, title in zip(axes[0], column_titles, strict=True):
        axis.set_title(title, fontsize=9)

    figure.suptitle(
        "Test examples from strongest decrease to strongest increase",
        fontsize=16,
    )
    figure.subplots_adjust(left=0.09, right=0.995, top=0.975, bottom=0.045, wspace=0.04, hspace=0.08)
    no2_colorbar_axis = figure.add_axes((0.10, 0.018, 0.39, 0.008))
    residual_colorbar_axis = figure.add_axes((0.535, 0.018, 0.285, 0.008))
    target_colorbar_axis = figure.add_axes((0.845, 0.018, 0.145, 0.008))
    figure.colorbar(no2_image, cax=no2_colorbar_axis, orientation="horizontal").set_label("NO2")
    figure.colorbar(residual_image, cax=residual_colorbar_axis, orientation="horizontal").set_label(
        "Transport residual"
    )
    figure.colorbar(target_image, cax=target_colorbar_axis, orientation="horizontal").set_label(
        "Contribution"
    )
    _save(figure, run_dir, "joint_prediction_samples")
