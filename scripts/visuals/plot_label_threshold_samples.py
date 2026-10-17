"""Plot eligible change-class counts across emissions change thresholds."""

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from preprocessing.stratify_plants import (
    AVG_NOX_END_COL,
    AVG_NOX_START_COL,
    build_stratification_candidates,
)

from config import STRATIFICATION_WINDOW_THRESHOLD

plt.switch_backend("Agg")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_STEM = REPOSITORY_ROOT / "assets" / "paper" / "visuals" / "label-threshold-samples"
DEFAULT_MAX_THRESHOLD = 1_000.0
DEFAULT_THRESHOLD_STEP = 10.0
DECREASE_COLOR = "#0072B2"
INCREASE_COLOR = "#E69F00"


def parse_args() -> argparse.Namespace:
    """Parse label-threshold plotting options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--max-threshold", type=float, default=DEFAULT_MAX_THRESHOLD)
    parser.add_argument("--threshold-step", type=float, default=DEFAULT_THRESHOLD_STEP)
    return parser.parse_args()


def _load_eligible_changes() -> np.ndarray:
    # Rebuild the exact metadata-eligible pool before class balancing
    with tempfile.TemporaryDirectory(prefix="label-threshold-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        candidates = build_stratification_candidates(
            temporary_path / "aoi-characteristics.png",
            temporary_path / "aoi-characteristics.csv",
            temporary_path / "aoi-selection.csv",
        )
    changes = candidates[AVG_NOX_END_COL].to_numpy() - candidates[AVG_NOX_START_COL].to_numpy()
    return changes[np.isfinite(changes)]


def _threshold_counts(
    changes: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Count change examples without applying split-specific class sampling
    ordered_changes = np.sort(changes)
    decrease_counts = np.searchsorted(ordered_changes, -thresholds, side="right")
    increase_counts = len(ordered_changes) - np.searchsorted(ordered_changes, thresholds, side="left")
    return decrease_counts, increase_counts


def _annotate_selected_threshold(
    axis: plt.Axes,
    selected_threshold: float,
    decrease_count: int,
    increase_count: int,
) -> None:
    # Mark the configured cutoff and its two eligible class counts
    axis.axvline(
        selected_threshold,
        color="#4D4D4D",
        linestyle="--",
        linewidth=1.0,
    )
    axis.annotate(
        f"{selected_threshold:g} lb/hr",
        (selected_threshold, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        color="#4D4D4D",
        fontsize=7,
        ha="left",
        va="top",
    )
    axis.scatter(selected_threshold, decrease_count, color=DECREASE_COLOR, s=18, zorder=3)
    axis.scatter(selected_threshold, increase_count, color=INCREASE_COLOR, s=18, zorder=3)
    axis.annotate(
        f"Decrease: {decrease_count:,}",
        (selected_threshold, decrease_count),
        xytext=(6, 7),
        textcoords="offset points",
        color=DECREASE_COLOR,
        fontsize=7,
        ha="left",
        va="bottom",
    )
    axis.annotate(
        f"Increase: {increase_count:,}",
        (selected_threshold, increase_count),
        xytext=(6, 7),
        textcoords="offset points",
        color=INCREASE_COLOR,
        fontsize=7,
        ha="left",
        va="bottom",
    )


def _plot_counts(
    thresholds: np.ndarray,
    decrease_counts: np.ndarray,
    increase_counts: np.ndarray,
    output_stem: Path,
) -> None:
    # Render the compact paper figure in PDF and PNG formats
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(3.5, 2.15))
    axis.plot(thresholds, decrease_counts, color=DECREASE_COLOR, linewidth=1.6, label="Decrease")
    axis.plot(thresholds, increase_counts, color=INCREASE_COLOR, linewidth=1.6, label="Increase")

    selected_index = int(np.flatnonzero(thresholds == STRATIFICATION_WINDOW_THRESHOLD)[0])
    _annotate_selected_threshold(
        axis,
        STRATIFICATION_WINDOW_THRESHOLD,
        int(decrease_counts[selected_index]),
        int(increase_counts[selected_index]),
    )
    axis.set(
        xlabel=r"NO$_x$ emissions change threshold (lb/hr)",
        ylabel="Eligible samples",
        xlim=(float(thresholds[0]), float(thresholds[-1])),
        ylim=(0, None),
    )
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Generate the label-threshold sample-count figure."""
    args = parse_args()
    thresholds = np.unique(
        np.append(
            np.arange(0.0, args.max_threshold + args.threshold_step, args.threshold_step),
            STRATIFICATION_WINDOW_THRESHOLD,
        )
    )
    changes = _load_eligible_changes()
    decrease_counts, increase_counts = _threshold_counts(changes, thresholds)
    _plot_counts(thresholds, decrease_counts, increase_counts, args.output_stem)

    selected_index = int(np.flatnonzero(thresholds == STRATIFICATION_WINDOW_THRESHOLD)[0])
    print(
        f"Eligible records: {len(changes):,}; at {STRATIFICATION_WINDOW_THRESHOLD:g} lb/hr: "
        f"decrease={decrease_counts[selected_index]:,}, increase={increase_counts[selected_index]:,}"
    )
    print(f"Wrote {args.output_stem.with_suffix('.pdf')} and {args.output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
