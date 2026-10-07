from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPOSITORY_ROOT / "assets" / "paper" / "visuals"
OUTPUT_STEM = OUTPUT_DIR / "classification-performance"

MODEL_NAMES = ("Baseline MLP", "ConvGRU")
MODEL_COLORS = ("#0072B2", "#E69F00")
MODEL_HATCHES = (None, "///")

CLASS_NAMES = ("Decrease", "Steady", "Increase")
SPLIT_NAMES = ("Train", "Validation", "Test")
PER_CLASS_F1 = np.array(
    [
        [[0.546, 0.361, 0.675], [0.436, 0.388, 0.635], [0.397, 0.382, 0.702]],
        [[0.614, 0.514, 0.722], [0.532, 0.580, 0.743], [0.444, 0.486, 0.684]],
    ]
)
PER_CLASS_RECALL = np.array(
    [
        [[0.548, 0.312, 0.745], [0.563, 0.332, 0.654], [0.457, 0.310, 0.771]],
        [[0.619, 0.474, 0.762], [0.612, 0.516, 0.776], [0.529, 0.460, 0.671]],
    ]
)
AGGREGATE_NAMES = ("Macro-F1", "Balanced acc.")
AGGREGATE_METRICS = np.array(
    [
        [
            [0.527, PER_CLASS_RECALL[0, 0].mean()],
            [0.486, PER_CLASS_RECALL[0, 1].mean()],
            [0.494, PER_CLASS_RECALL[0, 2].mean()],
        ],
        [
            [0.617, PER_CLASS_RECALL[1, 0].mean()],
            [0.619, PER_CLASS_RECALL[1, 1].mean()],
            [0.538, PER_CLASS_RECALL[1, 2].mean()],
        ],
    ]
)


def _plot_grouped_bars(
    axis: plt.Axes,
    category_names: tuple[str, ...],
    values: np.ndarray,
) -> None:
    category_count = len(category_names)
    group_gap = 0.8
    split_starts = np.arange(len(SPLIT_NAMES)) * (category_count + group_gap)
    positions = np.concatenate(
        [split_start + np.arange(category_count) for split_start in split_starts]
    )
    bar_width = 0.38

    for model_index, (model_name, color, hatch) in enumerate(
        zip(MODEL_NAMES, MODEL_COLORS, MODEL_HATCHES, strict=True)
    ):
        offset = (model_index - 0.5) * bar_width
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
        axis.bar_label(bars, fmt="%.3f", padding=2 + 5 * model_index, fontsize=6.5)

    axis.set_xticks(positions, category_names * len(SPLIT_NAMES))
    axis.set_ylim(0.0, 0.85)
    axis.set_yticks(np.arange(0.0, 0.81, 0.2))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    for split_index, (split_name, split_start) in enumerate(zip(SPLIT_NAMES, split_starts, strict=True)):
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
            axis.axvline(split_start - group_gap / 2, color="#BFBFBF", linewidth=0.6)


def main() -> None:
    """Generate classification performance figures for the paper."""
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

    figure, axes = plt.subplots(3, 1, figsize=(7.15, 5.45), sharey=True)
    panels = (
        ("(a) Per-class F1", CLASS_NAMES, PER_CLASS_F1),
        ("(b) Per-class recall", CLASS_NAMES, PER_CLASS_RECALL),
        ("(c) Aggregate metrics", AGGREGATE_NAMES, AGGREGATE_METRICS),
    )

    for axis, (title, category_names, values) in zip(axes, panels, strict=True):
        _plot_grouped_bars(axis, category_names, values)
        axis.set_title(title, pad=7)

    for axis in axes:
        axis.set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=2.2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    figure.savefig(f"{OUTPUT_STEM}.png", dpi=450, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
