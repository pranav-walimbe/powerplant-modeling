"""Generate the full-width metric-unit JGR version of paper Figure 4."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.visuals.common.plot_sample_construction import generate  # noqa: E402

OUTPUT_STEM = REPOSITORY_ROOT / "assets" / "paper" / "visuals" / "jgr" / "sample-construction"
POUNDS_TO_KILOGRAMS = 0.45359237


if __name__ == "__main__":
    generate(
        OUTPUT_STEM,
        figure_size=(6.5, 2.10),
        nox_delta_scale=POUNDS_TO_KILOGRAMS,
        nox_delta_units="kg/h",
        nox_delta_decimals=1,
    )
