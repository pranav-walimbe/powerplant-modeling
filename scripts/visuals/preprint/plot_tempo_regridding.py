"""Generate the preprint version of paper Figure 1."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.visuals.common.plot_tempo_regridding import (  # noqa: E402
    RegriddingLayout,
    generate,
)

OUTPUT_STEM = REPOSITORY_ROOT / "assets" / "paper" / "visuals" / "preprint" / "tempo-regridding"
LAYOUT = RegriddingLayout(
    figure_size=(6.0, 2.90),
    width_ratios=(1.0, 0.14, 1.0),
    height_ratios=(1.0, 0.18, 0.06),
    colorbar_row=2,
    colorbar_width_ratios=(0.2, 0.6, 0.2),
    horizontal_space=0.02,
    vertical_space=0.08,
    native_title="(a) Native TEMPO Level 2 footprints",
    grid_title="(b) Area-weighted 3 km grid",
    title_pad=5,
    process_text_y=0.59,
    arrow_start_x=0.22,
    arrow_end_x=0.78,
    arrow_y=0.39,
    arrow_linewidth=1.15,
    arrow_mutation_scale=10,
    colorbar_font_size=7.0,
    colorbar_label_pad=4,
    colorbar_tick_font_size=6.5,
    colorbar_tick_length=2.5,
    colorbar_tick_pad=3.5,
    left=0.085,
    right=0.985,
    top=0.91,
    bottom=0.15,
)


if __name__ == "__main__":
    generate(OUTPUT_STEM, LAYOUT)
