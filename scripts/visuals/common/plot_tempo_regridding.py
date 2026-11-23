"""Shared data loading and rendering for TEMPO regridding figures."""

from dataclasses import dataclass
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

EXAMPLE_AOI_ID = 9
EXAMPLE_LONGITUDE = -106.375
EXAMPLE_LATITUDE = 31.7569
EXAMPLE_GRANULE = Path("2023/08/TEMPO_NO2_L2_V04_20230804T160722Z_S004G08.nc")
NO2_SCALE = 1.0e15


@dataclass(frozen=True)
class RegriddingLayout:
    """Paper-specific layout values for the shared regridding renderer."""

    figure_size: tuple[float, float]
    width_ratios: tuple[float, float, float]
    height_ratios: tuple[float, ...]
    colorbar_row: int
    colorbar_width_ratios: tuple[float, float, float]
    horizontal_space: float
    vertical_space: float
    native_title: str
    grid_title: str
    title_pad: float
    process_text_y: float
    arrow_start_x: float
    arrow_end_x: float
    arrow_y: float
    arrow_linewidth: float
    arrow_mutation_scale: float
    colorbar_font_size: float
    colorbar_label_pad: float
    colorbar_tick_font_size: float
    colorbar_tick_length: float
    colorbar_tick_pad: float
    left: float
    right: float
    top: float
    bottom: float


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


def _format_map_axis(axis: plt.Axes) -> None:
    # Keep both panels in the same physical coordinate frame
    half_extent = IMG_RANGE / 2
    axis.set(
        xlim=(-half_extent, half_extent),
        ylim=(-half_extent, half_extent),
        aspect="equal",
        xticks=(),
        yticks=(),
    )
    axis.spines[["top", "right"]].set_visible(False)


def generate(output_stem: Path, layout: RegriddingLayout) -> None:
    """Generate PDF and PNG regridding figures with a supplied paper layout."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.0,
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

    figure = plt.figure(figsize=layout.figure_size)
    grid_layout = figure.add_gridspec(
        len(layout.height_ratios),
        3,
        width_ratios=layout.width_ratios,
        height_ratios=layout.height_ratios,
        hspace=layout.vertical_space,
        wspace=layout.horizontal_space,
    )
    colorbar_layout = grid_layout[layout.colorbar_row, :].subgridspec(
        1,
        3,
        width_ratios=layout.colorbar_width_ratios,
    )
    native_axis = figure.add_subplot(grid_layout[0, 0])
    process_axis = figure.add_subplot(grid_layout[0, 1])
    grid_axis = figure.add_subplot(grid_layout[0, 2])
    colorbar_axis = figure.add_subplot(colorbar_layout[0, 1])

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
    native_axis.set_title(layout.native_title, pad=layout.title_pad)
    _format_map_axis(native_axis)

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
    grid_axis.set_title(layout.grid_title, pad=layout.title_pad)
    _format_map_axis(grid_axis)

    process_axis.set_axis_off()
    process_axis.text(
        0.50,
        layout.process_text_y,
        "area-overlap\nweighted mean",
        ha="center",
        va="center",
        fontsize=7.0,
    )
    process_axis.annotate(
        "",
        xy=(layout.arrow_end_x, layout.arrow_y),
        xytext=(layout.arrow_start_x, layout.arrow_y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#303030",
            "linewidth": layout.arrow_linewidth,
            "mutation_scale": layout.arrow_mutation_scale,
        },
    )

    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(
        r"NO$_2$ column ($10^{15}$ molecules cm$^{-2}$)",
        fontsize=layout.colorbar_font_size,
        labelpad=layout.colorbar_label_pad,
    )
    colorbar.ax.tick_params(
        labelsize=layout.colorbar_tick_font_size,
        length=layout.colorbar_tick_length,
        pad=layout.colorbar_tick_pad,
    )
    colorbar.outline.set_linewidth(0.5)

    figure.subplots_adjust(
        left=layout.left,
        right=layout.right,
        top=layout.top,
        bottom=layout.bottom,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    figure.savefig(output_stem.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)
