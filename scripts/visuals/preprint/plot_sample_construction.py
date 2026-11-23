"""Generate the preprint version of the sample-construction paper figure."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.visuals.common.plot_sample_construction import generate  # noqa: E402

OUTPUT_STEM = REPOSITORY_ROOT / "assets" / "paper" / "visuals" / "preprint" / "sample-construction"


if __name__ == "__main__":
    generate(
        OUTPUT_STEM,
        figure_size=(7.15, 2.32),
        nox_delta_scale=1.0,
        nox_delta_units="lb/hr",
        nox_delta_decimals=1,
    )
