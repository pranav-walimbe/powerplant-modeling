"""Generate the full-width metric-unit JGR version of paper Figure 3."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.visuals.common.plot_no2_class_agreement_examples import generate  # noqa: E402

OUTPUT_STEM = (
    REPOSITORY_ROOT / "assets" / "paper" / "visuals" / "jgr" / "no2-class-agreement-examples"
)
POUNDS_TO_KILOGRAMS = 0.45359237


if __name__ == "__main__":
    generate(
        OUTPUT_STEM,
        figure_size=(6.5, 3.10),
        nox_delta_scale=POUNDS_TO_KILOGRAMS,
        nox_delta_units="kg/h",
        nox_delta_decimals=1,
        colorbar_gap_ratio=0.009,
    )
