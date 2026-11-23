"""Generate the wide metric-unit JGR version of paper Figure 2."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.visuals.common.plot_label_threshold_samples import generate  # noqa: E402

OUTPUT_STEM = REPOSITORY_ROOT / "assets" / "paper" / "visuals" / "jgr" / "label-threshold-samples"
MAX_THRESHOLD_KG_PER_HOUR = 350.0
THRESHOLD_STEP_KG_PER_HOUR = 5.0
POUNDS_TO_KILOGRAMS = 0.45359237


if __name__ == "__main__":
    generate(
        OUTPUT_STEM,
        max_threshold=MAX_THRESHOLD_KG_PER_HOUR,
        threshold_step=THRESHOLD_STEP_KG_PER_HOUR,
        value_scale=POUNDS_TO_KILOGRAMS,
        threshold_units="kg/h",
        threshold_decimals=1,
        figure_size=(6.5, 1.90),
        x_tick_step=50.0,
    )
