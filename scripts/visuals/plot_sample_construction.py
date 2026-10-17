"""Generate the complete training-input sequence used as paper Figure 2."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from modeling.dataset import NormalizationStats, NOxDataset
from modeling.train import _bilinear_interpolation_fill
from modeling.transport import TransportConfig, TransportTransform

from config import MODEL_CLASS_NAMES, MODEL_IMAGE_KEYS, RUNS_DIR

plt.switch_backend("Agg")


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPOSITORY_ROOT / "assets" / "paper" / "visuals"
OUTPUT_STEM = OUTPUT_DIR / "sample-construction"
RUN_DIR = Path(RUNS_DIR) / "classification_only_zscore_20261007_202105"

EXAMPLE_TRAIN_INDEX = 27_621
EXAMPLE_AOI_ID = 6_041
NO2_DISPLAY_SCALE = 1.0e15
WIND_ARROW_STEP = 4
FIGURE_WIDTH = 7.15
FIGURE_HEIGHT = 2.32


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NOxDataset("train", stats)
    image, _, _, label_index, _ = dataset[EXAMPLE_TRAIN_INDEX]
    image = image.unsqueeze(0)

    batch_size, timesteps, channels, height, width = image.shape
    frames = image.reshape(batch_size * timesteps, channels, height, width)
    completed_no2 = _bilinear_interpolation_fill(frames)
    completed = frames.clone()
    completed[:, :1] = completed_no2
    completed = completed.reshape(batch_size, timesteps, channels, height, width).float()

    image_center = torch.as_tensor(stats.image_center, dtype=completed.dtype)
    image_scale = torch.as_tensor(stats.image_scale, dtype=completed.dtype)
    numeric_channels = len(MODEL_IMAGE_KEYS)
    physical = (
        completed[:, :, :numeric_channels] * image_scale[None, None, :, None, None]
        + image_center[None, None, :, None, None]
    )
    classifier_input = completed.to(device)

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
    row = dataset.frame.iloc[EXAMPLE_TRAIN_INDEX]
    if int(row["aoi_id"]) != EXAMPLE_AOI_ID:
        raise ValueError(f"Training row {EXAMPLE_TRAIN_INDEX:,} is not AOI {EXAMPLE_AOI_ID}")
    return (
        physical[0].numpy(),
        residuals[0, :, 0].cpu().numpy(),
        MODEL_CLASS_NAMES[int(label_index)],
        float(row["avg_nox_end"] - row["avg_nox_start"]),
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


def _add_group_title(axis: plt.Axes, title: str) -> None:
    # Reserve a dedicated row for the modality heading
    axis.set_axis_off()
    axis.text(
        0.5,
        0.82,
        title,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="normal",
    )


def _add_timestep_label(axis: plt.Axes, label: str, *, fontsize: float = 6.3) -> None:
    # Reserve a separate row for each timestep label
    axis.set_axis_off()
    axis.text(0.5, 0.45, label, ha="center", va="center", fontsize=fontsize)


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
    layout = slot.subgridspec(
        6,
        5,
        height_ratios=(0.22, 0.13, 0.06, 1.0, 0.10, 0.045),
        hspace=0.0,
        wspace=0.06,
    )
    _add_group_title(figure.add_subplot(layout[0, :]), title)
    for timestep in range(5):
        _add_timestep_label(figure.add_subplot(layout[1, timestep]), f"$t_{timestep}$")
        axis = figure.add_subplot(layout[3, timestep])
        axis.imshow(values[timestep], origin="lower", cmap=color_map, norm=normalization)
        _format_raster_axis(axis)
    _add_colorbar(
        figure,
        figure.add_subplot(layout[5, 1:4]),
        normalization,
        color_map,
        colorbar_label,
    )


def _plot_wind_sequence(
    figure: plt.Figure,
    slot: plt.SubplotSpec,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
) -> None:
    # Use color for speed and fixed-length arrows for direction
    speed = np.hypot(wind_u, wind_v)
    direction_u = np.divide(wind_u, speed, out=np.zeros_like(wind_u), where=speed > 0)
    direction_v = np.divide(wind_v, speed, out=np.zeros_like(wind_v), where=speed > 0)
    normalization = _robust_normalization(speed)
    layout = slot.subgridspec(
        6,
        5,
        height_ratios=(0.22, 0.13, 0.06, 1.0, 0.10, 0.045),
        hspace=0.0,
        wspace=0.06,
    )
    _add_group_title(figure.add_subplot(layout[0, :]), "(b) 80 m wind speed and direction")
    arrow_slice = np.s_[WIND_ARROW_STEP // 2 :: WIND_ARROW_STEP, WIND_ARROW_STEP // 2 :: WIND_ARROW_STEP]
    arrow_y, arrow_x = np.mgrid[
        WIND_ARROW_STEP // 2 : wind_u.shape[-2] : WIND_ARROW_STEP,
        WIND_ARROW_STEP // 2 : wind_u.shape[-1] : WIND_ARROW_STEP,
    ]
    for timestep in range(5):
        _add_timestep_label(figure.add_subplot(layout[1, timestep]), f"$t_{timestep}$")
        axis = figure.add_subplot(layout[3, timestep])
        axis.imshow(speed[timestep], origin="lower", cmap="magma", norm=normalization)
        axis.quiver(
            arrow_x,
            arrow_y,
            direction_u[timestep][arrow_slice],
            direction_v[timestep][arrow_slice],
            color="white",
            angles="xy",
            scale_units="xy",
            scale=0.43,
            width=0.010,
            headwidth=3.0,
            pivot="middle",
        )
        _format_raster_axis(axis)
    _add_colorbar(
        figure,
        figure.add_subplot(layout[5, 1:4]),
        normalization,
        "magma",
        r"80 m wind speed (m s$^{-1}$)",
    )


def _plot_residual_sequence(
    figure: plt.Figure,
    slot: plt.SubplotSpec,
    residuals: np.ndarray,
) -> None:
    # Plot the four physical transitions with one shared scale
    display_values = residuals / NO2_DISPLAY_SCALE
    normalization = _residual_normalization(display_values)
    layout = slot.subgridspec(
        6,
        4,
        height_ratios=(0.22, 0.13, 0.06, 1.0, 0.10, 0.045),
        hspace=0.0,
        wspace=0.06,
    )
    _add_group_title(figure.add_subplot(layout[0, :]), "(d) Transport residuals")
    for interval in range(4):
        _add_timestep_label(
            figure.add_subplot(layout[1, interval]),
            f"$t_{interval} \\to t_{interval + 1}$",
            fontsize=5.6,
        )
        axis = figure.add_subplot(layout[3, interval])
        axis.imshow(display_values[interval], origin="lower", cmap="RdBu_r", norm=normalization)
        _format_raster_axis(axis)
    _add_colorbar(
        figure,
        figure.add_subplot(layout[5, 1:3]),
        normalization,
        "RdBu_r",
        r"Transport residual ($10^{15}$ molecules cm$^{-2}$)",
    )


def _add_sample_target(figure: plt.Figure, label: str, nox_delta: float) -> None:
    # Place the target and its determining change in the figure header
    header_axis = figure.add_axes((0.0, 0.895, 1.0, 0.090), zorder=20)
    header_axis.set_axis_off()
    header_axis.text(
        0.500,
        0.50,
        f"Behavior Class: {label.capitalize()} ($\\Delta$NO$_x$: {nox_delta:+.1f} lb/hr)",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="normal",
        color="#202020",
    )


def _plot_temporal_features(figure: plt.Figure) -> None:
    # Show the four cyclic context inputs as a compact symbolic vector
    axis = figure.add_axes((0.800, 0.170, 0.195, 0.630))
    axis.set_axis_off()
    axis.set_title("(e) Tabular temporal\nfeatures", fontsize=7.2, fontweight="normal", pad=4)
    bracket_color = "#4A4A4A"
    bracket_width = 0.9
    bracket_bottom = 0.37
    bracket_top = 0.61
    axis.plot((0.06, 0.06), (bracket_bottom, bracket_top), color=bracket_color, linewidth=bracket_width)
    axis.plot((0.06, 0.10), (bracket_top, bracket_top), color=bracket_color, linewidth=bracket_width)
    axis.plot((0.06, 0.10), (bracket_bottom, bracket_bottom), color=bracket_color, linewidth=bracket_width)
    axis.plot((0.94, 0.94), (bracket_bottom, bracket_top), color=bracket_color, linewidth=bracket_width)
    axis.plot((0.90, 0.94), (bracket_top, bracket_top), color=bracket_color, linewidth=bracket_width)
    axis.plot((0.90, 0.94), (bracket_bottom, bracket_bottom), color=bracket_color, linewidth=bracket_width)
    feature_labels = (
        "sin(local solar hour)",
        "cos(local solar hour)",
        "sin(day of year)",
        "cos(day of year)",
    )
    for position, feature_label in zip((0.575, 0.520, 0.465, 0.410), feature_labels, strict=True):
        axis.text(0.50, position, feature_label, ha="center", va="center", fontsize=5.8)


def main() -> None:
    """Generate PDF and PNG versions of the complete sample figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "mathtext.fontset": "dejavusans",
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
        right=0.780,
        bottom=0.125,
        top=0.835,
        hspace=0.58,
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
    _plot_temporal_features(figure)
    _add_sample_target(figure, label, nox_delta)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=450)
    plt.close(figure)


if __name__ == "__main__":
    main()
