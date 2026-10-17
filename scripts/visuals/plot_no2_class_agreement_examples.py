"""Compare clear and unclear visual agreement in two increase samples."""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from modeling.dataset import NormalizationStats, NOxDataset
from modeling.train import _bilinear_interpolation_fill

from config import MODEL_IMAGE_KEYS, RUNS_DIR

plt.switch_backend("Agg")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPOSITORY_ROOT / "assets" / "paper" / "visuals"
OUTPUT_STEM = OUTPUT_DIR / "no2-class-agreement-examples"
RUN_DIR = Path(RUNS_DIR) / "classification_only_zscore_20261007_202105"
NO2_DISPLAY_SCALE = 1.0e15


@dataclass(frozen=True)
class Example:
    """Fixed training example and its agreement label."""

    row_index: int
    aoi_id: int
    agreement: str


EXAMPLES = (
    Example(25_169, 3_612, "Clear visual class agreement"),
    Example(10_790, 1_912, "Unclear visual class agreement"),
)


def _load_examples() -> tuple[list[np.ndarray], list[float]]:
    # Load both records and bilinearly fill missing NO2 pixels
    with (RUN_DIR / "normalization_stats.json").open() as source:
        stats = NormalizationStats.from_dict(json.load(source))
    dataset = NOxDataset("train", stats)
    sequences: list[np.ndarray] = []
    nox_deltas: list[float] = []
    for example in EXAMPLES:
        image, _, _, _, _ = dataset[example.row_index]
        row = dataset.frame.iloc[example.row_index]
        if int(row["aoi_id"]) != example.aoi_id:
            raise ValueError(f"Training row {example.row_index:,} is not AOI {example.aoi_id}")
        completed_no2 = _bilinear_interpolation_fill(image)
        physical_no2 = (
            completed_no2[:, 0].numpy() * stats.image_scale[MODEL_IMAGE_KEYS.index("no2")]
            + stats.image_center[MODEL_IMAGE_KEYS.index("no2")]
        )
        sequences.append(physical_no2 / NO2_DISPLAY_SCALE)
        nox_deltas.append(float(row["avg_nox_end"] - row["avg_nox_start"]))
    return sequences, nox_deltas


def _normalization(sequences: list[np.ndarray]) -> colors.Normalize:
    # Share robust limits across both five-timestep examples
    minimum, maximum = np.percentile(np.concatenate(sequences), (2.0, 98.0))
    return colors.Normalize(vmin=float(minimum), vmax=float(maximum))


def _format_raster_axis(axis: plt.Axes) -> None:
    # Remove ticks while retaining a subtle raster boundary
    axis.set_xticks(())
    axis.set_yticks(())
    for spine in axis.spines.values():
        spine.set_color("#AFAFAF")
        spine.set_linewidth(0.35)


def _plot_example(
    figure: plt.Figure,
    slot: plt.SubplotSpec,
    panel_label: str,
    example: Example,
    sequence: np.ndarray,
    nox_delta: float,
    normalization: colors.Normalize,
) -> None:
    # Plot one labeled five-timestep NO2 sequence
    layout = slot.subgridspec(
        4,
        5,
        height_ratios=(0.23, 0.14, 1.0, 0.055),
        hspace=0.04,
        wspace=0.06,
    )
    header_axis = figure.add_subplot(layout[0, :])
    header_axis.set_axis_off()
    header_axis.text(
        0.0,
        0.5,
        f"({panel_label}) {example.agreement} ($\\Delta$NO$_x$: {nox_delta:+.1f} lb/hr)",
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="normal",
    )

    image = None
    for timestep in range(sequence.shape[0]):
        timestep_axis = figure.add_subplot(layout[1, timestep])
        timestep_axis.set_axis_off()
        timestep_axis.text(0.5, 0.45, f"$t_{timestep}$", ha="center", va="center", fontsize=6.2)
        raster_axis = figure.add_subplot(layout[2, timestep])
        image = raster_axis.imshow(
            sequence[timestep],
            origin="lower",
            cmap="viridis",
            norm=normalization,
            interpolation="nearest",
        )
        _format_raster_axis(raster_axis)

    colorbar = figure.colorbar(image, cax=figure.add_subplot(layout[3, 1:4]), orientation="horizontal")
    colorbar.set_label(r"NO$_2$ column ($10^{15}$ molecules cm$^{-2}$)", fontsize=5.5, labelpad=0.4)
    colorbar.ax.tick_params(labelsize=5.0, length=1.6, pad=1)
    colorbar.outline.set_linewidth(0.35)


def main() -> None:
    """Generate the two-example NO2 sequence comparison."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sequences, nox_deltas = _load_examples()
    normalization = _normalization(sequences)
    figure = plt.figure(figsize=(3.5, 2.85))
    layout = figure.add_gridspec(
        2,
        1,
        left=0.035,
        right=0.990,
        bottom=0.075,
        top=0.985,
        hspace=0.36,
    )
    for panel_label, slot, example, sequence, nox_delta in zip(
        ("a", "b"),
        layout,
        EXAMPLES,
        sequences,
        nox_deltas,
        strict=True,
    ):
        _plot_example(
            figure,
            slot,
            panel_label,
            example,
            sequence,
            nox_delta,
            normalization,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
