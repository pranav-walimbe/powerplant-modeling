"""Generate the preprint version of the NO2 class-agreement figure."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.visuals.common.plot_no2_class_agreement_examples import generate  # noqa: E402

OUTPUT_STEM = (
    REPOSITORY_ROOT
    / "assets"
    / "paper"
    / "visuals"
    / "preprint"
    / "no2-class-agreement-examples"
)


if __name__ == "__main__":
    generate(
        OUTPUT_STEM,
        figure_size=(3.5, 2.85),
        nox_delta_scale=1.0,
        nox_delta_units="lb/hr",
        nox_delta_decimals=1,
        colorbar_gap_ratio=0.0,
    )
