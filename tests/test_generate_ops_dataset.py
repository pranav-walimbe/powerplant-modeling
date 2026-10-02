"""Tests for operations evaluation dataset selection."""

import unittest
from datetime import datetime, timedelta, timezone

import polars as pl
from preprocessing.generate_ops_dataset import build_ops_candidates, filter_eligible_aois


class GenerateOpsDatasetTest(unittest.TestCase):
    def test_canonical_event_samples_and_aoi_minimum(self) -> None:
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        sample_rows = (
            (0, 10, "increase", 0, 2.1, 4, "increase-near.npz"),
            (1, 10, "increase", 0, 3.5, 4, "increase-far.npz"),
            (2, 10, "decrease", 5, 7.1, 9, "decrease.npz"),
            (3, 10, "steady", 10, 12.1, 14, "steady.npz"),
            (4, 20, "increase", 0, 2.2, 4, "other-increase.npz"),
        )
        samples = pl.DataFrame(
            {
                "sample_id": [row[0] for row in sample_rows],
                "aoi_id": [row[1] for row in sample_rows],
                "delta_category": [row[2] for row in sample_rows],
                "t0_timestamp": [base + timedelta(hours=row[3]) for row in sample_rows],
                "t2_timestamp": [base + timedelta(hours=row[4]) for row in sample_rows],
                "t4_timestamp": [base + timedelta(hours=row[5]) for row in sample_rows],
                "raster_bundle_path": [row[6] for row in sample_rows],
                "source_split": ["test"] * len(sample_rows),
            }
        )
        transitions = pl.DataFrame(
            {
                "aoi_id": [10, 10, 10, 20],
                "facilityId": [1, 1, 1, 2],
                "emissions_hour_utc": [
                    base + timedelta(hours=2),
                    base + timedelta(hours=7),
                    base + timedelta(hours=12),
                    base + timedelta(hours=2),
                ],
                "event_direction": ["start", "stop", "start", "start"],
            }
        )

        candidates = build_ops_candidates(samples, transitions)
        selected = filter_eligible_aois(candidates, min_sample_count=1)

        self.assertEqual(
            selected.sort("delta_category")["raster_bundle_path"].to_list(),
            ["decrease.npz", "increase-near.npz", "steady.npz"],
        )
        self.assertEqual(selected["aoi_id"].unique().to_list(), [10])


if __name__ == "__main__":
    unittest.main()
