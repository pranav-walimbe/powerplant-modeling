from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPOSITORY_ROOT / "assets" / "paper" / "visuals"
OUTPUT_STEM = OUTPUT_DIR / "classification-performance"
TEST_OUTPUT_STEM = OUTPUT_DIR / "classification-performance-test"

MODEL_NAMES = ("Baseline MLP", "ConvGRU")
MODEL_COLORS = ("#0072B2", "#E69F00")
MODEL_HATCHES = (None, "///")

CLASS_NAMES = ("Decrease", "Steady", "Increase")
SPLIT_NAMES = ("Train", "Validation", "Test")
PER_CLASS_F1 = np.array(
    [
        [[0.552, 0.385, 0.654], [0.451, 0.421, 0.702], [0.386, 0.397, 0.579]],
        [[0.613, 0.528, 0.719], [0.511, 0.562, 0.739], [0.449, 0.539, 0.689]],
    ]
)
PER_CLASS_RECALL = np.array(
    [
        [[0.574, 0.359, 0.674], [0.524, 0.358, 0.745], [0.527, 0.377, 0.539]],
        [[0.597, 0.500, 0.760], [0.600, 0.512, 0.749], [0.471, 0.525, 0.694]],
    ]
)
AGGREGATE_NAMES = ("Macro-F1",)
AGGREGATE_METRICS = np.array(
    [
        [
            [0.530],
            [0.525],
            [0.454],
        ],
        [
            [0.620],
            [0.604],
            [0.559],
        ],
    ]
)


def _plot_grouped_bars(
    axis: plt.Axes,
    category_names: tuple[str, ...],
    values: np.ndarray,
    split_names: tuple[str, ...],
    *,
    show_split_names: bool = True,
) -> None:
    category_count = len(category_names)
    group_gap = 0.8
    split_starts = np.arange(len(split_names)) * (category_count + group_gap)
    positions = np.concatenate(
        [split_start + np.arange(category_count) for split_start in split_starts]
    )
    bar_width = 0.30
    model_offset = 0.25

    for model_index, (model_name, color, hatch) in enumerate(
        zip(MODEL_NAMES, MODEL_COLORS, MODEL_HATCHES, strict=True)
    ):
        offset = -model_offset if model_index == 0 else model_offset
        bars = axis.bar(
            positions + offset,
            values[model_index].reshape(-1),
            bar_width,
            color=color,
            edgecolor="white",
            hatch=hatch,
            label=model_name,
            linewidth=0.6,
        )
        for bar in bars:
            axis.annotate(
                f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    axis.set_xticks(positions, category_names * len(split_names))
    axis.set_ylim(0.0, 0.85)
    axis.set_yticks(np.arange(0.0, 0.81, 0.2))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    if not show_split_names:
        return

    for split_index, (split_name, split_start) in enumerate(zip(split_names, split_starts, strict=True)):
        split_center = split_start + (category_count - 1) / 2
        axis.text(
            split_center,
            -0.28,
            split_name,
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            transform=axis.get_xaxis_transform(),
        )
        if split_index:
            previous_split_end = split_starts[split_index - 1] + category_count - 1
            divider_position = (previous_split_end + split_start) / 2
            axis.axvline(divider_position, color="#BFBFBF", linewidth=0.6)


def _save_all_splits_figure() -> None:
    # Retain the original three-row layout for direct comparison across splits
    figure, axes = plt.subplots(3, 1, figsize=(7.15, 5.45), sharey=True)
    panels = (
        ("(a) Per-class F1", CLASS_NAMES, PER_CLASS_F1),
        ("(b) Per-class recall", CLASS_NAMES, PER_CLASS_RECALL),
        ("(c) Macro-F1", AGGREGATE_NAMES, AGGREGATE_METRICS),
    )

    for axis, (title, category_names, values) in zip(axes, panels, strict=True):
        _plot_grouped_bars(axis, category_names, values, SPLIT_NAMES)
        axis.set_title(title, pad=7)

    for axis in axes:
        axis.set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=2.2)

    figure.savefig(f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    figure.savefig(f"{OUTPUT_STEM}.png", dpi=450, bbox_inches="tight")
    plt.close(figure)


def _save_test_figure() -> None:
    # The final split index is the held-out test set
    test_index = len(SPLIT_NAMES) - 1
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), sharey=True)
    panels = (
        ("(a) Per-class F1", CLASS_NAMES, PER_CLASS_F1[:, test_index : test_index + 1]),
        ("(b) Per-class recall", CLASS_NAMES, PER_CLASS_RECALL[:, test_index : test_index + 1]),
        ("(c) Macro-F1", AGGREGATE_NAMES, AGGREGATE_METRICS[:, test_index : test_index + 1]),
    )

    for axis, (title, category_names, values) in zip(axes, panels, strict=True):
        _plot_grouped_bars(
            axis,
            category_names,
            values,
            (SPLIT_NAMES[test_index],),
            show_split_names=False,
        )
        axis.set_title(title, pad=7)

    axes[0].set_ylabel("Test score")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86), w_pad=1.4)

    figure.savefig(f"{TEST_OUTPUT_STEM}.pdf", bbox_inches="tight")
    figure.savefig(f"{TEST_OUTPUT_STEM}.png", dpi=450, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Generate all-splits and test-only classification figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_all_splits_figure()
    _save_test_figure()


if __name__ == "__main__":
    main()
