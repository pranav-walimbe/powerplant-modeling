"""Generate the real TEMPO footprint-to-grid transformation used as paper Figure 1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib import colors
from matplotlib.collections import PolyCollection
from preprocessing.regrid import (
    GOOD_QUALITY_FLAG,
    AoiGrid,
    GranulePixels,
    build_granule_spatial_index,
    read_granule_pixels,
    regrid_aoi_raster,
)
from preprocessing.stratify_utils import WGS84_TO_CONUS

from config import IMG_RANGE, MIN_PIXEL_CLOUD, TEMPO_DIR

plt.switch_backend("Agg")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPOSITORY_ROOT / "assets" / "paper" / "visuals"
OUTPUT_STEM = OUTPUT_DIR / "tempo-regridding"

EXAMPLE_AOI_ID = 9
EXAMPLE_LONGITUDE = -106.375
EXAMPLE_LATITUDE = 31.7569
EXAMPLE_GRANULE = Path("2023/08/TEMPO_NO2_L2_V04_20230804T160722Z_S004G08.nc")
NO2_SCALE = 1.0e15


def _load_example() -> tuple[AoiGrid, GranulePixels, np.ndarray]:
    # Reproduce the selected real scan with the production regridding path
    grid = AoiGrid.from_lon_lat(EXAMPLE_AOI_ID, EXAMPLE_LONGITUDE, EXAMPLE_LATITUDE)
    pixels = read_granule_pixels(str(Path(TEMPO_DIR) / EXAMPLE_GRANULE))
    selected = build_granule_spatial_index(pixels).select_grid(grid)
    raster = regrid_aoi_raster(selected, grid)
    return grid, selected, raster.no2


def _project_footprints(grid: AoiGrid, pixels: GranulePixels) -> tuple[np.ndarray, np.ndarray]:
    # Express native footprint corners as kilometres from the grid centre
    corner_x, corner_y = WGS84_TO_CONUS.transform(pixels.corner_longitudes, pixels.corner_latitudes)
    vertices = np.stack(
        (
            (corner_x - grid.x_m) / 1000.0,
            (corner_y - grid.y_m) / 1000.0,
        ),
        axis=2,
    )
    polygons = shapely.polygons(np.stack((corner_x, corner_y), axis=2))
    accepted = (
        shapely.is_valid(polygons)
        & (shapely.area(polygons) > 0)
        & (pixels.quality_flags == GOOD_QUALITY_FLAG)
        & (pixels.cloud_fractions <= MIN_PIXEL_CLOUD)
        & np.isfinite(pixels.values)
    )
    return vertices, accepted


def _format_map_axis(axis: plt.Axes, *, show_y_label: bool) -> None:
    # Keep both panels in the same physical coordinate frame
    half_extent = IMG_RANGE / 2
    axis.set(
        xlim=(-half_extent, half_extent),
        ylim=(-half_extent, half_extent),
        aspect="equal",
        xticks=(-half_extent, 0, half_extent),
        yticks=(-half_extent, 0, half_extent),
        xlabel="East-west distance (km)",
    )
    if show_y_label:
        axis.set_ylabel("North-south distance (km)")
    axis.tick_params(labelsize=6.5, length=2.5)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    """Generate PDF and PNG versions of the real-data regridding figure."""
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
    grid, pixels, gridded = _load_example()
    vertices, accepted = _project_footprints(grid, pixels)
    accepted_values = pixels.values[accepted] / NO2_SCALE
    normalization = colors.Normalize(
        vmin=float(np.percentile(accepted_values, 2)),
        vmax=float(np.percentile(accepted_values, 98)),
    )
    color_map = plt.colormaps["viridis"]

    figure = plt.figure(figsize=(6.0, 2.90))
    layout = figure.add_gridspec(
        3,
        3,
        width_ratios=(1.0, 0.24, 1.0),
        height_ratios=(1.0, 0.18, 0.06),
        hspace=0.08,
        wspace=0.10,
    )
    native_axis = figure.add_subplot(layout[0, 0])
    process_axis = figure.add_subplot(layout[0, 1])
    grid_axis = figure.add_subplot(layout[0, 2])
    colorbar_axis = figure.add_subplot(layout[2, :])

    face_colors = np.tile(colors.to_rgba("#D9D9D9"), (len(vertices), 1))
    face_colors[accepted] = color_map(normalization(accepted_values))
    native_axis.add_collection(
        PolyCollection(
            vertices,
            facecolors=face_colors,
            edgecolors=np.where(
                accepted[:, None],
                colors.to_rgba("white"),
                colors.to_rgba("#7F7F7F"),
            ),
            linewidths=0.20,
            rasterized=True,
        )
    )
    grid_edges = np.linspace(-IMG_RANGE / 2, IMG_RANGE / 2, grid.size + 1)
    for edge in grid_edges:
        native_axis.axvline(edge, color="white", linewidth=0.20, alpha=0.42, zorder=3)
        native_axis.axhline(edge, color="white", linewidth=0.20, alpha=0.42, zorder=3)
    native_axis.set_facecolor("#D9D9D9")
    native_axis.set_title("(a) Native TEMPO Level 2 footprints", pad=5)
    _format_map_axis(native_axis, show_y_label=True)

    half_extent = IMG_RANGE / 2
    image = grid_axis.imshow(
        gridded / NO2_SCALE,
        cmap=color_map,
        norm=normalization,
        origin="upper",
        extent=(-half_extent, half_extent, -half_extent, half_extent),
        interpolation="none",
    )
    grid_axis.set_facecolor("#D9D9D9")
    grid_axis.set_title("(b) Area-weighted 3 km grid", pad=5)
    _format_map_axis(grid_axis, show_y_label=False)

    process_axis.set_axis_off()
    process_axis.annotate(
        "area-overlap\nweighted mean",
        xy=(0.96, 0.50),
        xytext=(0.04, 0.50),
        ha="center",
        va="center",
        fontsize=7.2,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#303030",
            "linewidth": 1.35,
            "mutation_scale": 12,
        },
    )

    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(r"NO$_2$ column ($10^{15}$ molecules cm$^{-2}$)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5, length=2.5)
    colorbar.outline.set_linewidth(0.5)

    figure.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.15)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
