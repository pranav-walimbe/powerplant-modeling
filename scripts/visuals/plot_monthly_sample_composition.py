"""Plot monthly model accuracy with sample size and class composition."""

import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import MODEL_CLASS_NAMES

plt.switch_backend("Agg")


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_STEM = REPOSITORY_ROOT / "assets" / "paper" / "visuals" / "monthly-sample-composition"
RUN_DIR = Path("/global/home/users/pranavwalimbe/model_runs/classification_only_zscore_20261007_202105")

MODEL_NAMES = ("Baseline MLP", "ConvGRU")
MODEL_FILES = ("seasonal_test_predictions.csv", "test_predictions.csv")
MODEL_COLORS = ("#0072B2", "#E69F00")
MODEL_MARKERS = ("o", "s")
MODEL_LINESTYLES = ("--", "-")

FINAL_TIMESTAMP_COLUMN = "t4_timestamp"
TRUE_CLASS_COLUMN = "class_true"
PREDICTED_CLASS_COLUMN = "class_predicted"
PREDICTION_COLUMNS = (FINAL_TIMESTAMP_COLUMN, TRUE_CLASS_COLUMN, PREDICTED_CLASS_COLUMN)

CLASS_COLORS = ("#E68A8A", "#A6A6A6", "#7BAED4")
FIGURE_SIZE = (3.35, 5.4)
RIGHT_MARGIN_MONTHS = 0.25


def _load_predictions() -> dict[str, pd.DataFrame]:
    # Load both models on the same held-out records
    frames = {
        model_name: pd.read_csv(RUN_DIR / filename, usecols=PREDICTION_COLUMNS)
        for model_name, filename in zip(MODEL_NAMES, MODEL_FILES, strict=True)
    }
    reference = frames[MODEL_NAMES[0]][[FINAL_TIMESTAMP_COLUMN, TRUE_CLASS_COLUMN]]
    for model_name in MODEL_NAMES[1:]:
        if not reference.equals(frames[model_name][[FINAL_TIMESTAMP_COLUMN, TRUE_CLASS_COLUMN]]):
            raise ValueError(f"{model_name} predictions are not aligned with {MODEL_NAMES[0]}")
    return frames


def _monthly_values(
    frames: dict[str, pd.DataFrame],
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    # Calculate model accuracy and true-class counts for each month
    reference = frames[MODEL_NAMES[0]]
    months = pd.to_datetime(reference[FINAL_TIMESTAMP_COLUMN], utc=True).dt.month
    month_numbers = sorted(months.unique())
    accuracies = {
        model_name: np.asarray(
            [
                (frame.loc[months == month, TRUE_CLASS_COLUMN] == frame.loc[months == month, PREDICTED_CLASS_COLUMN])
                .mean()
                for month in month_numbers
            ],
            dtype=np.float64,
        )
        for model_name, frame in frames.items()
    }
    counts = np.stack(
        [
            np.bincount(
                reference.loc[months == month, TRUE_CLASS_COLUMN].to_numpy(dtype=np.int64),
                minlength=len(MODEL_CLASS_NAMES),
            )
            for month in month_numbers
        ]
    )
    return [calendar.month_abbr[month] for month in month_numbers], accuracies, counts


def _plot_monthly_accuracy(
    axis: plt.Axes,
    month_labels: list[str],
    accuracies: dict[str, np.ndarray],
) -> None:
    # Compare both models across calendar months
    positions = np.arange(len(month_labels))
    for model_name, color, marker, linestyle in zip(
        MODEL_NAMES,
        MODEL_COLORS,
        MODEL_MARKERS,
        MODEL_LINESTYLES,
        strict=True,
    ):
        axis.plot(
            positions,
            accuracies[model_name],
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            marker=marker,
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=model_name,
        )
    axis.set(
        title="(a) Accuracy by month",
        ylabel="Accuracy",
        xticks=positions,
        xticklabels=month_labels,
        ylim=(0.25, 0.72),
        yticks=np.arange(0.3, 0.71, 0.1),
    )
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)


def _plot_sample_composition(
    axis: plt.Axes,
    month_labels: list[str],
    counts: np.ndarray,
) -> None:
    # Stack true-class counts within each month
    totals = counts.sum(axis=1)
    positions = np.arange(len(month_labels))
    bottom = np.zeros(len(month_labels), dtype=np.float64)
    for class_index, (class_name, color) in enumerate(zip(MODEL_CLASS_NAMES, CLASS_COLORS, strict=True)):
        values = counts[:, class_index]
        axis.bar(
            positions,
            values,
            bottom=bottom,
            width=0.78,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=class_name.title(),
        )
        bottom += values

    axis.set(
        title="(b) Sample size and class composition by month",
        xlabel="Month",
        ylabel="Samples",
        xticks=positions,
        xticklabels=month_labels,
        ylim=(0, totals.max() * 1.04),
    )
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False)


def _style_axis(axis: plt.Axes) -> None:
    # Apply the shared paper styling
    axis.margins(x=0.015)
    left_limit, right_limit = axis.get_xlim()
    axis.set_xlim(left_limit, right_limit + RIGHT_MARGIN_MONTHS)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    """Generate the single-column monthly performance and composition figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 6.5,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    frames = _load_predictions()
    month_labels, accuracies, counts = _monthly_values(frames)
    figure, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE)
    _plot_monthly_accuracy(axes[0], month_labels, accuracies)
    _plot_sample_composition(axes[1], month_labels, counts)
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout(h_pad=3.2)

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
