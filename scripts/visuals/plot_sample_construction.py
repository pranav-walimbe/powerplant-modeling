"""Generate the complete training-input sequence used as paper Figure 2."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from matplotlib.lines import Line2D
from modeling.dataset import NormalizationStats, NOxDataset
from modeling.train import _load_masked_checkpoint
from modeling.transport import TransportConfig, TransportTransform

from config import MODEL_CLASS_NAMES, MODEL_IMAGE_KEYS, RUNS_DIR

plt.switch_backend("Agg")


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPOSITORY_ROOT / "assets" / "paper" / "visuals"
OUTPUT_STEM = OUTPUT_DIR / "sample-construction"
RUN_DIR = Path(RUNS_DIR) / "classification_only_zscore_20261006_205047"

EXAMPLE_TRAIN_INDEX = 27_621
NO2_DISPLAY_SCALE = 1.0e15
WIND_ARROW_STEP = 4
FIGURE_WIDTH = 7.15
FIGURE_HEIGHT = 2.90


def _load_json(path: Path) -> dict[str, object]:
    # Load one JSON metadata object
    with path.open() as source:
        return json.load(source)


def _transport_config(metadata: dict[str, object]) -> TransportConfig:
    # Restore the physical transport settings recorded by training
    lifetime = float(metadata["decay_lifetime_hours"]) if metadata["decay_enabled"] else None
    return TransportConfig(
        decay_lifetime_hours=lifetime,
        dispersion_sigma_pixels_at_one_hour=float(metadata["dispersion_sigma_pixels_at_one_hour"]),
        next_wind_weight=float(metadata["next_wind_weight"]),
        wind_speed_multiplier=float(metadata["wind_speed_multiplier"]),
    )


def _load_example() -> tuple[np.ndarray, np.ndarray, str, float]:
    # Recreate EDA example 4 through the recorded model-input pipeline
    run_config = _load_json(RUN_DIR / "run_config.json")
    stats = NormalizationStats.from_dict(_load_json(RUN_DIR / "normalization_stats.json"))
    checkpoint_path = Path(str(run_config["pretrained_masked_model_weights"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, model = _load_masked_checkpoint(checkpoint_path, device)
    masked_stats = checkpoint["normalization_stats"]
    dataset = NOxDataset("train", stats, source_image_stats=masked_stats)
    image, _, _, label_index, _ = dataset[EXAMPLE_TRAIN_INDEX]
    image = image.unsqueeze(0).to(device)

    batch_size, timesteps, channels, height, width = image.shape
    frames = image.reshape(batch_size * timesteps, channels, height, width)
    with torch.inference_mode():
        completed_no2 = model.fill_missing(frames)
    completed = frames.clone()
    completed[:, :1] = completed_no2
    completed = completed.reshape(batch_size, timesteps, channels, height, width).float()

    masked_center = torch.as_tensor(masked_stats["image_center"], dtype=completed.dtype, device=device)
    masked_scale = torch.as_tensor(masked_stats["image_scale"], dtype=completed.dtype, device=device)
    numeric_channels = len(MODEL_IMAGE_KEYS)
    physical = (
        completed[:, :, :numeric_channels] * masked_scale[None, None, :, None, None]
        + masked_center[None, None, :, None, None]
    )

    classifier_center = torch.as_tensor(stats.image_center, dtype=completed.dtype, device=device)
    classifier_scale = torch.as_tensor(stats.image_scale, dtype=completed.dtype, device=device)
    classifier_input = completed.clone()
    classifier_input[:, :, :numeric_channels] = (
        physical - classifier_center[None, None, :, None, None]
    ) / classifier_scale[None, None, :, None, None]

    transport_metadata = run_config["transport"]
    transport = TransportTransform(
        stats.image_center,
        stats.image_scale,
        config=_transport_config(transport_metadata),
    )
    elapsed_hours = torch.from_numpy(dataset.elapsed_hours[[EXAMPLE_TRAIN_INDEX]]).to(device)
    no2_center_scale = torch.as_tensor(
        ((stats.image_center[0], stats.image_scale[0]),),
        dtype=completed.dtype,
        device=device,
    )
    with torch.inference_mode():
        residuals, _ = transport.physical_residual(
            classifier_input,
            elapsed_hours,
            no2_center_scale,
        )
    return (
        physical[0].cpu().numpy(),
        residuals[0, :, 0].cpu().numpy(),
        MODEL_CLASS_NAMES[int(label_index)],
        float(
            dataset.frame.iloc[EXAMPLE_TRAIN_INDEX]["avg_nox_end"]
            - dataset.frame.iloc[EXAMPLE_TRAIN_INDEX]["avg_nox_start"]
        ),
    )


def _robust_normalization(values: np.ndarray) -> colors.Normalize:
    # Share one robust color scale across each input sequence
    minimum, maximum = np.nanpercentile(values, (2.0, 98.0))
    return colors.Normalize(vmin=float(minimum), vmax=float(maximum))


def _residual_normalization(values: np.ndarray) -> colors.TwoSlopeNorm:
    # Center the transport-residual color scale on zero
    limit = float(np.nanpercentile(np.abs(values), 98.0))
    return colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def _format_raster_axis(axis: plt.Axes) -> None:
    # Remove ticks while retaining a subtle raster boundary
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#AFAFAF")
        spine.set_linewidth(0.35)


def _add_group_title(figure: plt.Figure, slot: plt.SubplotSpec, title: str) -> None:
    # Place the group title independently of timestep labels
    bounds = slot.get_position(figure)
    figure.text(
        (bounds.x0 + bounds.x1) / 2,
        bounds.y1 + 0.008,
        title,
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="normal",
    )


def _add_colorbar(
    figure: plt.Figure,
    axis: plt.Axes,
    normalization: colors.Normalize,
    color_map: str,
    label: str,
) -> None:
    # Add a compact horizontal scale below one sequence
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=color_map),
        cax=axis,
        orientation="horizontal",
    )
    colorbar.set_label(label, fontsize=5.8, labelpad=0.5)
    colorbar.ax.tick_params(labelsize=5.3, length=1.8, pad=1)
    colorbar.outline.set_linewidth(0.35)


def _plot_scalar_sequence(
    figure: plt.Figure,
    slot: plt.SubplotSpec,
    values: np.ndarray,
    *,
    title: str,
    color_map: str,
    colorbar_label: str,
    normalization: colors.Normalize,
) -> None:
    # Plot five aligned rasters with one shared scale
    layout = slot.subgridspec(2, 5, height_ratios=(1.0, 0.08), hspace=0.24, wspace=0.06)
    for timestep in range(5):
        axis = figure.add_subplot(layout[0, timestep])
        axis.imshow(values[timestep], origin="lower", cmap=color_map, norm=normalization)
        axis.set_title(f"$t_{timestep}$", fontsize=6.3, pad=1.0)
        _format_raster_axis(axis)
    _add_colorbar(
        figure,
        figure.add_subplot(layout[1, :]),
        normalization,
        color_map,
        colorbar_label,
    )
    _add_group_title(figure, slot, title)


def _plot_wind_sequence(
    figure: plt.Figure,
    slot: plt.SubplotSpec,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
) -> None:
    # Combine wind magnitude and direction in each timestep raster
    speed = np.hypot(wind_u, wind_v)
    normalization = _robust_normalization(speed)
    layout = slot.subgridspec(2, 5, height_ratios=(1.0, 0.08), hspace=0.24, wspace=0.06)
    arrow_slice = np.s_[WIND_ARROW_STEP // 2 :: WIND_ARROW_STEP, WIND_ARROW_STEP // 2 :: WIND_ARROW_STEP]
    arrow_y, arrow_x = np.mgrid[
        WIND_ARROW_STEP // 2 : wind_u.shape[-2] : WIND_ARROW_STEP,
        WIND_ARROW_STEP // 2 : wind_u.shape[-1] : WIND_ARROW_STEP,
    ]
    for timestep in range(5):
        axis = figure.add_subplot(layout[0, timestep])
        axis.imshow(speed[timestep], origin="lower", cmap="magma", norm=normalization)
        axis.quiver(
            arrow_x,
            arrow_y,
            wind_u[timestep][arrow_slice],
            wind_v[timestep][arrow_slice],
            color="white",
            angles="xy",
            scale_units="xy",
            scale=3.5,
            width=0.010,
            headwidth=3.0,
        )
        axis.set_title(f"$t_{timestep}$", fontsize=6.3, pad=1.0)
        _format_raster_axis(axis)
    _add_colorbar(
        figure,
        figure.add_subplot(layout[1, :]),
        normalization,
        "magma",
        r"80 m wind speed (m s$^{-1}$)",
    )
    _add_group_title(figure, slot, "(b) 80 m wind speed and direction")


def _plot_residual_sequence(
    figure: plt.Figure,
    slot: plt.SubplotSpec,
    residuals: np.ndarray,
) -> None:
    # Plot the four physical transitions with one shared scale
    display_values = residuals / NO2_DISPLAY_SCALE
    normalization = _residual_normalization(display_values)
    layout = slot.subgridspec(2, 4, height_ratios=(1.0, 0.08), hspace=0.24, wspace=0.06)
    for interval in range(4):
        axis = figure.add_subplot(layout[0, interval])
        axis.imshow(display_values[interval], origin="lower", cmap="RdBu_r", norm=normalization)
        axis.set_title(f"$t_{interval} \\to t_{interval + 1}$", fontsize=6.0, pad=1.0)
        _format_raster_axis(axis)
    _add_colorbar(
        figure,
        figure.add_subplot(layout[1, :]),
        normalization,
        "RdBu_r",
        r"Transport residual ($10^{15}$ molecules cm$^{-2}$)",
    )
    _add_group_title(figure, slot, "(d) Transport residuals")


def _add_behavior_label(figure: plt.Figure, label: str, nox_delta: float) -> None:
    # Group the complete input matrix with its behavior target
    bracket_x = 0.825
    bracket_left = 0.810
    bracket_bottom = 0.080
    bracket_top = 0.940
    line_style = {
        "color": "#4A4A4A",
        "linewidth": 1.15,
        "solid_capstyle": "round",
        "transform": figure.transFigure,
    }
    figure.add_artist(Line2D((bracket_x, bracket_x), (bracket_bottom, bracket_top), **line_style))
    figure.add_artist(Line2D((bracket_left, bracket_x), (bracket_top, bracket_top), **line_style))
    figure.add_artist(Line2D((bracket_left, bracket_x), (bracket_bottom, bracket_bottom), **line_style))
    figure.text(
        0.910,
        0.535,
        "Behavior label:",
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="normal",
    )
    figure.text(
        0.910,
        0.475,
        label.upper(),
        ha="center",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color={"decrease": "#2B6CB0", "steady": "#555555", "increase": "#C53030"}[label],
    )
    figure.text(
        0.910,
        0.405,
        f"NO$_x$ delta:\n{nox_delta:+.1f} lb/hr",
        ha="center",
        va="top",
        fontsize=7.4,
        fontweight="normal",
        color="#202020",
    )


def main() -> None:
    """Generate PDF and PNG versions of the complete sample figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    physical, residuals, label, nox_delta = _load_example()
    no2 = physical[:, MODEL_IMAGE_KEYS.index("no2")] / NO2_DISPLAY_SCALE
    temperature = physical[:, MODEL_IMAGE_KEYS.index("temperature_2m_k")]
    wind_u = physical[:, MODEL_IMAGE_KEYS.index("wind_u_80m_mps")]
    wind_v = physical[:, MODEL_IMAGE_KEYS.index("wind_v_80m_mps")]

    figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    layout = figure.add_gridspec(
        2,
        2,
        left=0.025,
        right=0.795,
        bottom=0.075,
        top=0.94,
        hspace=0.50,
        wspace=0.10,
    )
    _plot_scalar_sequence(
        figure,
        layout[0, 0],
        no2,
        title="(a) NO$_2$ observations",
        color_map="viridis",
        colorbar_label=r"NO$_2$ column ($10^{15}$ molecules cm$^{-2}$)",
        normalization=_robust_normalization(no2),
    )
    _plot_wind_sequence(figure, layout[0, 1], wind_u, wind_v)
    _plot_scalar_sequence(
        figure,
        layout[1, 0],
        temperature,
        title="(c) 2 m surface temperature",
        color_map="coolwarm",
        colorbar_label="2 m temperature (K)",
        normalization=_robust_normalization(temperature),
    )
    _plot_residual_sequence(figure, layout[1, 1], residuals)
    _add_behavior_label(figure, label, nox_delta)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=450)
    plt.close(figure)


if __name__ == "__main__":
    main()
