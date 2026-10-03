"""Build spatial source-contribution targets from stratification metadata."""

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from config import IMG_SIZE, MODEL_TARGET_COL, SEQUENCE_TIMESTEPS

CONTRIBUTION_CHANNEL_NAMES = ("decrease", "increase")
CONTRIBUTION_TARGET_SIZE = 12
SOURCE_PIXEL_NOX_COLUMNS = tuple(f"t{index}_source_pixel_nox" for index in range(SEQUENCE_TIMESTEPS))


class ContributionTargetDataset(Protocol):
    """Dataset fields required to materialize contribution targets."""

    frame: pd.DataFrame
    elapsed_hours: np.ndarray

    def __len__(self) -> int:
        """Return the number of records."""
        ...


def _source_records(payload: object) -> list[dict[str, float | int]]:
    # Decode one nullable CSV JSON value
    if payload is None or (isinstance(payload, float) and math.isnan(payload)):
        return []
    if isinstance(payload, str):
        return json.loads(payload)
    return list(payload)


def _facility_changes(
    timestep_payloads: Sequence[object],
    elapsed_hours: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Align facilities across timesteps and calculate late-minus-early changes
    facility_positions: dict[int, tuple[float, float]] = {}
    facility_nox: dict[int, np.ndarray] = {}
    for timestep, payload in enumerate(timestep_payloads):
        for source in _source_records(payload):
            facility_id = int(source["facility_id"])
            facility_positions.setdefault(facility_id, (float(source["row"]), float(source["col"])))
            facility_nox.setdefault(facility_id, np.zeros(SEQUENCE_TIMESTEPS, dtype=np.float64))[timestep] = float(
                source["nox"]
            )

    rows = np.empty(len(facility_nox), dtype=np.float64)
    columns = np.empty(len(facility_nox), dtype=np.float64)
    changes = np.empty(len(facility_nox), dtype=np.float64)
    for output_index, (facility_id, values) in enumerate(facility_nox.items()):
        interval_means = (values[:-1] + values[1:]) / 2.0
        early = np.average(interval_means[:2], weights=elapsed_hours[:2])
        late = np.average(interval_means[2:], weights=elapsed_hours[2:])
        rows[output_index], columns[output_index] = facility_positions[facility_id]
        changes[output_index] = late - early
    return rows, columns, changes


def _target_coordinate(source_coordinate: np.ndarray, target_size: int) -> np.ndarray:
    # Preserve pixel-center geometry while changing grid resolution
    return (source_coordinate + 0.5) * target_size / IMG_SIZE - 0.5


def _gaussian_splat(
    rows: np.ndarray,
    columns: np.ndarray,
    weights: np.ndarray,
    target_size: int,
    sigma: float,
) -> np.ndarray:
    # Preserve each source weight after clipping the Gaussian to the target grid
    target = np.zeros((target_size, target_size), dtype=np.float32)
    target_rows = _target_coordinate(rows, target_size)
    target_columns = _target_coordinate(columns, target_size)
    if sigma == 0.0:
        nearest_rows = np.clip(np.floor(target_rows + 0.5).astype(np.int64), 0, target_size - 1)
        nearest_columns = np.clip(np.floor(target_columns + 0.5).astype(np.int64), 0, target_size - 1)
        np.add.at(target, (nearest_rows, nearest_columns), weights.astype(np.float32))
        return target

    grid_rows, grid_columns = np.mgrid[:target_size, :target_size]
    for row, column, weight in zip(target_rows, target_columns, weights, strict=True):
        kernel = np.exp(-0.5 * ((grid_rows - row) ** 2 + (grid_columns - column) ** 2) / sigma**2)
        target += (weight * kernel / kernel.sum()).astype(np.float32)
    return target


def build_contribution_target(
    timestep_payloads: Sequence[object],
    elapsed_hours: np.ndarray,
    label: str,
    *,
    sigma: float,
    target_size: int = CONTRIBUTION_TARGET_SIZE,
) -> np.ndarray:
    """Build class-conditioned decrease and increase contribution rasters.

    Args:
        timestep_payloads: Five nullable source-pixel JSON values.
        elapsed_hours: Four durations between consecutive observation times.
        label: Stored decrease, steady, or increase class.
        sigma: Gaussian width in target-grid cells. Zero selects nearest-cell splatting.
        target_size: Output height and width.

    Returns:
        Two nonnegative rasters ordered as decrease then increase. The active
        non-steady channel sums to one and steady targets are all zero.
    """
    target = np.zeros((len(CONTRIBUTION_CHANNEL_NAMES), target_size, target_size), dtype=np.float32)
    if label == "steady":
        return target

    rows, columns, changes = _facility_changes(timestep_payloads, elapsed_hours)
    if label == "decrease":
        selected = changes < 0
        channel = CONTRIBUTION_CHANNEL_NAMES.index("decrease")
        weights = -changes[selected]
    else:
        selected = changes > 0
        channel = CONTRIBUTION_CHANNEL_NAMES.index("increase")
        weights = changes[selected]
    if weights.size == 0:
        return target

    weights = weights / weights.sum()
    target[channel] = _gaussian_splat(rows[selected], columns[selected], weights, target_size, sigma)
    return target


def materialize_contribution_targets(
    datasets: Mapping[str, ContributionTargetDataset],
    output_dir: str | Path,
    *,
    sigma: float,
    target_size: int = CONTRIBUTION_TARGET_SIZE,
) -> dict[str, Path]:
    """Materialize contribution targets for each geographic split.

    Args:
        datasets: Split datasets carrying metadata and timestep durations.
        output_dir: New directory for memory-mapped target arrays.
        sigma: Gaussian width in target-grid cells.
        target_size: Output height and width.

    Returns:
        Target array path for each split.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    target_paths: dict[str, Path] = {}
    for split, dataset in datasets.items():
        target_path = destination / f"{split}_contribution_targets.npy"
        target_array = np.lib.format.open_memmap(
            target_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(dataset), len(CONTRIBUTION_CHANNEL_NAMES), target_size, target_size),
        )
        payloads = dataset.frame.loc[:, SOURCE_PIXEL_NOX_COLUMNS].to_numpy(dtype=object)
        labels = dataset.frame[MODEL_TARGET_COL].to_numpy(dtype=str)
        for index, (row_payloads, elapsed_hours, label) in enumerate(
            zip(payloads, dataset.elapsed_hours, labels, strict=True)
        ):
            target_array[index] = build_contribution_target(
                row_payloads,
                elapsed_hours,
                label,
                sigma=sigma,
                target_size=target_size,
            )
        target_array.flush()
        del target_array
        target_paths[split] = target_path
        print(f"Contribution-target construction {split}: {len(dataset):,} records")
    return target_paths


def contribution_target_metadata(sigma: float, target_size: int = CONTRIBUTION_TARGET_SIZE) -> dict[str, object]:
    """Describe the persisted contribution-target contract.

    Args:
        sigma: Gaussian width in target-grid cells.
        target_size: Output height and width.

    Returns:
        Serializable target-construction metadata.
    """
    return {
        "target_size": target_size,
        "channels": list(CONTRIBUTION_CHANNEL_NAMES),
        "splat_sigma_cells": sigma,
        "coordinate_source_size": IMG_SIZE,
        "temporal_method": "late_minus_early_trapezoidal_facility_mean",
        "normalization": "active_class_facility_changes_sum_to_one",
        "inactive_class": "all_zero",
        "steady_target": "both_channels_all_zero",
        "source_columns": list(SOURCE_PIXEL_NOX_COLUMNS),
    }
