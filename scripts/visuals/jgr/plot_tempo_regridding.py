"""Generate the JGR version of paper Figure 1."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.visuals.common.plot_tempo_regridding import (  # noqa: E402
    RegriddingLayout,
    generate,
)

OUTPUT_STEM = REPOSITORY_ROOT / "assets" / "paper" / "visuals" / "jgr" / "tempo-regridding"
LAYOUT = RegriddingLayout(
    figure_size=(6.5, 2.65),
    width_ratios=(1.0, 0.16, 1.0),
    height_ratios=(1.0, 0.045),
    colorbar_row=1,
    colorbar_width_ratios=(0.22, 0.56, 0.22),
    horizontal_space=0.01,
    vertical_space=0.08,
    native_title="(a) Native TEMPO L2 footprints",
    grid_title="(b) Area-weighted 3 km grid",
    title_pad=3,
    process_text_y=0.61,
    arrow_start_x=0.14,
    arrow_end_x=0.86,
    arrow_y=0.41,
    arrow_linewidth=0.9,
    arrow_mutation_scale=8,
    colorbar_font_size=6.5,
    colorbar_label_pad=1,
    colorbar_tick_font_size=6.0,
    colorbar_tick_length=2.0,
    colorbar_tick_pad=1,
    left=0.015,
    right=0.995,
    top=0.88,
    bottom=0.15,
)


if __name__ == "__main__":
    generate(OUTPUT_STEM, LAYOUT)
