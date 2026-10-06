"""Generate the TEMPO footprint-to-grid schematic used as paper Figure 1."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as PolygonPatch
from shapely.geometry import Polygon, box

plt.switch_backend("Agg")


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPOSITORY_ROOT / "assets" / "paper" / "visuals"
OUTPUT_STEM = OUTPUT_DIR / "tempo-regridding"

AOI_HALF_WIDTH_KM = 36.0
GRID_SIZE = 24
GRID_CELL_KM = 3.0
PLANT_X_KM = -13.0
PLANT_Y_KM = -8.0
WIND_ANGLE_DEGREES = 22.0


@dataclass(frozen=True)
class Footprint:
    """One illustrative TEMPO footprint and its retrieval state."""

    polygon: Polygon
    no2_column: float
    is_valid: bool


def _plume_column(x_km: float, y_km: float) -> float:
    # Create a smooth downwind enhancement over a regional background
    angle = np.deg2rad(WIND_ANGLE_DEGREES)
    relative_x = x_km - PLANT_X_KM
    relative_y = y_km - PLANT_Y_KM
    downwind = relative_x * np.cos(angle) + relative_y * np.sin(angle)
    crosswind = -relative_x * np.sin(angle) + relative_y * np.cos(angle)
    plume = 5.6 * np.exp(-0.5 * ((downwind - 13.0) / 15.0) ** 2 - 0.5 * (crosswind / 5.0) ** 2)
    plume *= 1.0 / (1.0 + np.exp(-downwind / 2.0))
    source = 2.2 * np.exp(-0.5 * (relative_x / 3.4) ** 2 - 0.5 * (relative_y / 3.4) ** 2)
    regional = 0.35 * np.cos((x_km + y_km) / 18.0)
    return 2.2 + regional + plume + source


def _native_footprints() -> list[Footprint]:
    # Tile the AOI with skewed quadrilaterals that resemble native scan footprints
    rng = np.random.default_rng(42)
    footprints = []
    y_centers = np.arange(-37.0, 38.0, 3.0)
    x_centers = np.arange(-38.0, 39.0, 5.0)
    for row_index, y_center in enumerate(y_centers):
        row_offset = 1.1 * np.sin(row_index * 0.65)
        for x_center in x_centers + row_offset:
            center_jitter = rng.normal(0.0, 0.22, size=2)
            center_x = x_center + center_jitter[0]
            center_y = y_center + center_jitter[1]
            half_width = 2.8 + rng.normal(0.0, 0.12)
            half_height = 1.72 + rng.normal(0.0, 0.08)
            skew = 0.60 + rng.normal(0.0, 0.08)
            polygon = Polygon(
                [
                    (center_x - half_width - skew, center_y - half_height),
                    (center_x + half_width - skew, center_y - half_height),
                    (center_x + half_width + skew, center_y + half_height),
                    (center_x - half_width + skew, center_y + half_height),
                ]
            )
            cloud_rejected = ((center_x + 20.0) / 10.0) ** 2 + ((center_y - 20.0) / 8.0) ** 2 < 1.0
            quality_rejected = rng.random() < 0.025
            footprints.append(
                Footprint(
                    polygon=polygon,
                    no2_column=_plume_column(center_x, center_y) + rng.normal(0.0, 0.12),
                    is_valid=not (cloud_rejected or quality_rejected),
                )
            )
    return footprints


def _area_weighted_grid(footprints: list[Footprint]) -> tuple[np.ndarray, np.ndarray]:
    # Average accepted footprint values by their overlap with each target cell
    edges = np.linspace(-AOI_HALF_WIDTH_KM, AOI_HALF_WIDTH_KM, GRID_SIZE + 1)
    gridded = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float64)
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            cell = box(edges[column], edges[row], edges[column + 1], edges[row + 1])
            weighted_total = 0.0
            overlap_total = 0.0
            for footprint in footprints:
                if not footprint.is_valid or not cell.intersects(footprint.polygon):
                    continue
                overlap_area = cell.intersection(footprint.polygon).area
                weighted_total += overlap_area * footprint.no2_column
                overlap_total += overlap_area
            if overlap_total > 0.0:
                gridded[row, column] = weighted_total / overlap_total
    return edges, gridded


def _plot_source(axis: plt.Axes) -> None:
    # Mark the source location consistently in both panels
    axis.scatter(
        PLANT_X_KM,
        PLANT_Y_KM,
        marker="*",
        s=75,
        color="#202020",
        edgecolor="white",
        linewidth=0.55,
        zorder=8,
    )
    axis.annotate(
        "power plant",
        (PLANT_X_KM, PLANT_Y_KM),
        xytext=(-27.0, -16.0),
        fontsize=6.5,
        arrowprops={"arrowstyle": "-", "color": "#404040", "linewidth": 0.7},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        zorder=9,
    )
def _format_map_axis(axis: plt.Axes, *, show_y_label: bool) -> None:
    # Use identical map limits and restrained axes in both panels
    axis.set(
        xlim=(-AOI_HALF_WIDTH_KM, AOI_HALF_WIDTH_KM),
        ylim=(-AOI_HALF_WIDTH_KM, AOI_HALF_WIDTH_KM),
        aspect="equal",
        xticks=(-36, 0, 36),
        yticks=(-36, 0, 36),
    )
    if show_y_label:
        axis.set_ylabel("North-south distance (km)")
    axis.tick_params(labelsize=6.5, length=2.5)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    """Generate PDF and PNG versions of the regridding schematic."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    footprints = _native_footprints()
    edges, gridded = _area_weighted_grid(footprints)
    valid_values = np.asarray([footprint.no2_column for footprint in footprints if footprint.is_valid])
    normalization = colors.Normalize(vmin=np.percentile(valid_values, 3), vmax=np.percentile(valid_values, 98))
    color_map = plt.colormaps["viridis"]

    figure = plt.figure(figsize=(7.15, 3.35))
    layout = figure.add_gridspec(
        3,
        3,
        width_ratios=(1.0, 0.30, 1.0),
        height_ratios=(1.0, 0.14, 0.06),
        hspace=0.08,
        wspace=0.10,
    )
    native_axis = figure.add_subplot(layout[0, 0])
    process_axis = figure.add_subplot(layout[0, 1])
    grid_axis = figure.add_subplot(layout[0, 2])
    colorbar_axis = figure.add_subplot(layout[2, :])

    for footprint in footprints:
        if footprint.is_valid:
            face_color = color_map(normalization(footprint.no2_column))
            hatch = None
        else:
            face_color = "#D9D9D9"
            hatch = "////"
        native_axis.add_patch(
            PolygonPatch(
                np.asarray(footprint.polygon.exterior.coords),
                facecolor=face_color,
                edgecolor="white" if footprint.is_valid else "#7F7F7F",
                linewidth=0.28,
                hatch=hatch,
            )
        )
    for edge in edges:
        native_axis.axvline(edge, color="white", linewidth=0.28, alpha=0.45, zorder=6)
        native_axis.axhline(edge, color="white", linewidth=0.28, alpha=0.45, zorder=6)
    native_axis.set_title("(a) Native footprints over the target grid", pad=5)
    native_axis.legend(
        handles=[Patch(facecolor="#D9D9D9", edgecolor="#7F7F7F", hatch="////", label="quality rejected")],
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        fontsize=6.3,
        handlelength=1.6,
        borderpad=0.35,
    )
    native_axis.set_xlabel("East-west distance (km)")
    _plot_source(native_axis)
    _format_map_axis(native_axis, show_y_label=True)

    masked_grid = np.ma.masked_invalid(gridded)
    image = grid_axis.pcolormesh(
        edges,
        edges,
        masked_grid,
        cmap=color_map,
        norm=normalization,
        edgecolors=(1.0, 1.0, 1.0, 0.42),
        linewidth=0.24,
        shading="flat",
    )
    grid_axis.set_facecolor("#D9D9D9")
    grid_axis.set_title("(b) Area-weighted 3 km AOI grid", pad=5)
    grid_axis.set_xlabel("East-west distance (km)")
    grid_axis.text(
        34.0,
        32.5,
        "24 × 24 cells",
        ha="right",
        va="top",
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.8},
    )
    _plot_source(grid_axis)
    _format_map_axis(grid_axis, show_y_label=False)

    process_axis.set_axis_off()
    process_axis.annotate(
        "area-overlap\nweighted mean",
        xy=(0.95, 0.50),
        xytext=(0.05, 0.50),
        ha="center",
        va="center",
        fontsize=6.5,
        arrowprops={"arrowstyle": "-|>", "color": "#4D4D4D", "linewidth": 0.9},
    )

    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(r"NO$_2$ column ($10^{15}$ molecules cm$^{-2}$)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5, length=2.5)
    colorbar.outline.set_linewidth(0.5)

    figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.13)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    figure.savefig(f"{OUTPUT_STEM}.png", dpi=450, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
