"""Generate the preprint version of paper Figure 2."""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.visuals.common.plot_label_threshold_samples import generate  # noqa: E402

OUTPUT_STEM = REPOSITORY_ROOT / "assets" / "paper" / "visuals" / "preprint" / "label-threshold-samples"
DEFAULT_MAX_THRESHOLD = 1_000.0
DEFAULT_THRESHOLD_STEP = 10.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=OUTPUT_STEM)
    parser.add_argument("--max-threshold", type=float, default=DEFAULT_MAX_THRESHOLD)
    parser.add_argument("--threshold-step", type=float, default=DEFAULT_THRESHOLD_STEP)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    generate(
        arguments.output_stem,
        max_threshold=arguments.max_threshold,
        threshold_step=arguments.threshold_step,
        value_scale=1.0,
        threshold_units="lb/hr",
        threshold_decimals=0,
        figure_size=(3.5, 2.15),
    )
