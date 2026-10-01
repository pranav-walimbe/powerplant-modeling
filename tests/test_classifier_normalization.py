"""Regression tests for train-only classifier raster normalization."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from modeling.dataset import MODEL_FEATURE_NAMES, NormalizationStats, NOxDataset, compute_stats

from config import MODEL_IMAGE_KEYS, MODEL_MASK_KEYS, SEQUENCE_TIMESTEPS


def _record(raster_path: str, category: str = "steady") -> dict[str, object]:
    # Build the metadata columns consumed by the classifier dataset
    record: dict[str, object] = {
        "aoi_id": 1,
        "lon": -120.0,
        "raster_bundle_path": raster_path,
        "delta_category": category,
    }
    for index in range(SEQUENCE_TIMESTEPS):
        record[f"t{index}_timestamp"] = f"2025-01-01T{index + 12:02d}:00:00Z"
    return record


def _write_bundle(path: Path, offset: float) -> dict[str, np.ndarray]:
    # Write one five-frame raster bundle with known channel distributions
    base = np.arange(1, SEQUENCE_TIMESTEPS + 1, dtype=np.float32).reshape(-1, 1, 1)
    channels = {
        "no2": base + offset,
        "temperature_2m_k": 270.0 + base + offset,
        "wind_u_80m_mps": -5.0 + base + offset,
        "wind_v_80m_mps": 5.0 + base + offset,
        MODEL_MASK_KEYS[0]: np.ones_like(base, dtype=np.float32),
    }
    np.savez(path, **channels)
    return channels


class ClassifierNormalizationTest(unittest.TestCase):
    def test_compute_stats_fits_every_raster_channel_from_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataframe_dir = root / "dataframes"
            dataframe_dir.mkdir()
            first = _write_bundle(root / "first.npz", 0.0)
            second = _write_bundle(root / "second.npz", 5.0)
            pd.DataFrame([_record("first.npz"), _record("second.npz")]).to_csv(
                dataframe_dir / "train_df.csv",
                index=False,
            )

            stats = compute_stats(
                "train",
                dataset_dir=root,
                dataframe_dir=dataframe_dir,
                progress_interval=0,
            )

            expected_values = {
                name: np.concatenate((first[name].ravel(), second[name].ravel())) for name in MODEL_IMAGE_KEYS
            }
            no2 = expected_values["no2"]
            lower, median, upper = np.percentile(no2, (25, 50, 75))
            self.assertAlmostEqual(stats.image_center[0], median)
            self.assertAlmostEqual(stats.image_scale[0], (upper - lower) / 1.349)
            for channel, name in enumerate(MODEL_IMAGE_KEYS[1:], start=1):
                self.assertAlmostEqual(stats.image_center[channel], expected_values[name].mean())
                self.assertAlmostEqual(stats.image_scale[channel], expected_values[name].std())
            self.assertEqual(stats.image_valid_pixels, (10, 10, 10, 10))

    def test_completed_rasters_are_converted_from_masked_to_classifier_space(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataframe_dir = root / "dataframes"
            dataframe_dir.mkdir()
            pd.DataFrame([_record("unused.npz")]).to_csv(dataframe_dir / "train_df.csv", index=False)
            completed = np.full((1, SEQUENCE_TIMESTEPS, 5, 1, 1), 2.0, dtype=np.float32)
            completed[:, :, 4] = 1.0
            completed_path = root / "completed.npy"
            np.save(completed_path, completed)
            stats = NormalizationStats(
                image_keys=MODEL_IMAGE_KEYS,
                image_center=(100.0, 280.0, 0.0, 0.0),
                image_scale=(10.0, 2.0, 4.0, 5.0),
                image_valid_pixels=(1, 1, 1, 1),
                feature_names=MODEL_FEATURE_NAMES,
                feature_mean=(0.0, 0.0, 0.0, 0.0),
                feature_std=(1.0, 1.0, 1.0, 1.0),
                target_name="delta_category",
                training_records=1,
            )
            masked_stats = {
                "image_center": (50.0, 250.0, 10.0, -10.0),
                "image_scale": (5.0, 10.0, 2.0, 4.0),
            }

            dataset = NOxDataset(
                "train",
                stats,
                dataset_dir=root,
                dataframe_dir=dataframe_dir,
                completed_raster_path=completed_path,
                source_image_stats=masked_stats,
            )
            image = dataset[0][0].numpy()

            expected = np.asarray((-4.0, -5.0, 3.5, -0.4), dtype=np.float32)
            np.testing.assert_allclose(image[:, :4, 0, 0], np.broadcast_to(expected, (SEQUENCE_TIMESTEPS, 4)))
            np.testing.assert_array_equal(image[:, 4, 0, 0], np.ones(SEQUENCE_TIMESTEPS))
            np.testing.assert_array_equal(dataset.no2_normalization, np.asarray([[100.0, 10.0]]))


if __name__ == "__main__":
    unittest.main()
