"""Dataset loading and train-only normalization for TEMPO raster modeling."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from modeling.transport import TransportTransform
from torch.utils.data import Dataset

from config import (
    DATASET_DF,
    DATASET_DIR,
    IMG_SIZE,
    MODEL_CLASS_NAMES,
    MODEL_IMAGE_CLIP_ABS,
    MODEL_IMAGE_KEYS,
    MODEL_MASK_KEYS,
    MODEL_SEASONAL_FEATURES,
    MODEL_TARGET_COL,
    SEQUENCE_TIMESTEPS,
)

RASTER_PATH_COL = "raster_bundle_path"
NO2_CHANNEL = MODEL_IMAGE_KEYS.index("no2")
MIN_SCALE = 1e-12
DEGREES_PER_SOLAR_HOUR = 15.0
SECONDS_PER_DAY = 86_400.0
TIMESTEP_TIME_COLUMNS = tuple(f"t{index}_timestamp" for index in range(SEQUENCE_TIMESTEPS))
FINAL_TIMESTEP_TIME_COLUMN = TIMESTEP_TIME_COLUMNS[-1]
CONTRIBUTION_CHANNEL_NAMES = ("decrease", "increase")
CONTRIBUTION_TARGET_SIZE = IMG_SIZE
SOURCE_PIXEL_NOX_COLUMNS = tuple(f"t{index}_source_pixel_nox" for index in range(SEQUENCE_TIMESTEPS))

MODEL_FEATURE_NAMES = MODEL_SEASONAL_FEATURES


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


@dataclass(frozen=True)
class NormalizationStats:
    """JSON-safe train-split preprocessing state used by every data split."""

    image_keys: tuple[str, ...]
    image_center: tuple[float, ...]
    image_scale: tuple[float, ...]
    image_valid_pixels: tuple[int, ...]
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    target_name: str
    training_records: int

    def to_dict(self) -> dict[str, object]:
        """Return the normalization state as JSON-safe values.

        Returns:
            Serialized normalization fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "NormalizationStats":
        """Build normalization state from JSON-safe values.

        Args:
            values: Serialized normalization fields.

        Returns:
            Parsed normalization state.
        """
        return cls(
            image_keys=tuple(str(name) for name in values["image_keys"]),
            image_center=tuple(float(value) for value in values["image_center"]),
            image_scale=tuple(float(value) for value in values["image_scale"]),
            image_valid_pixels=tuple(int(value) for value in values["image_valid_pixels"]),
            feature_names=tuple(str(name) for name in values["feature_names"]),
            feature_mean=tuple(float(value) for value in values["feature_mean"]),
            feature_std=tuple(float(value) for value in values["feature_std"]),
            target_name=str(values.get("target_name", MODEL_TARGET_COL)),
            training_records=int(values["training_records"]),
        )


def _read_split_frame(split: str, dataframe_dir: Path) -> pd.DataFrame:
    # Load the split produced by dataset generation
    path = dataframe_dir / f"{split}_df.csv"
    return pd.read_csv(path)


def _midpoint_local_time(frame: pd.DataFrame) -> pd.Series:
    # Locate the sample midpoint and convert it from UTC to local solar time
    start = pd.to_datetime(frame[TIMESTEP_TIME_COLUMNS[0]], errors="coerce", utc=True)
    end = pd.to_datetime(frame[TIMESTEP_TIME_COLUMNS[-1]], errors="coerce", utc=True)
    utc_time = start + (end - start) / 2
    longitude = pd.to_numeric(frame["lon"], errors="coerce")
    return utc_time + pd.to_timedelta(longitude / DEGREES_PER_SOLAR_HOUR, unit="h")


def midpoint_local_solar_hour(frame: pd.DataFrame) -> pd.Series:
    """Calculate fractional local solar hour at each sample midpoint.

    Args:
        frame: Records containing sequence timestamps and longitude.

    Returns:
        Fractional hours in the range from zero through twenty-four.
    """
    solar_time = _midpoint_local_time(frame)
    return (
        solar_time.dt.hour
        + solar_time.dt.minute / 60.0
        + solar_time.dt.second / 3_600.0
        + solar_time.dt.microsecond / 3_600_000_000.0
    )


def _cyclic_feature_matrix(solar_time: pd.Series) -> np.ndarray:
    # Encode one local-solar timestamp as daily and annual phase
    seconds = (
        solar_time.dt.hour * 3_600
        + solar_time.dt.minute * 60
        + solar_time.dt.second
        + solar_time.dt.microsecond / 1_000_000
    ).to_numpy(dtype=np.float64)
    daily_angle = 2 * np.pi * seconds / SECONDS_PER_DAY
    annual_position = solar_time.dt.dayofyear.to_numpy(dtype=np.float64) - 1.0 + seconds / SECONDS_PER_DAY
    days_in_year = np.where(solar_time.dt.is_leap_year, 366.0, 365.0)
    annual_angle = 2 * np.pi * annual_position / days_in_year
    return np.column_stack(
        (
            np.sin(daily_angle),
            np.cos(daily_angle),
            np.sin(annual_angle),
            np.cos(annual_angle),
        )
    )


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    # Preserve midpoint context for the seasonal-only baseline
    return _cyclic_feature_matrix(_midpoint_local_time(frame))


def _raster_path(serialized_path: object, dataset_dir: Path) -> Path:
    # Resolve a stored raster path against the dataset root
    path = Path(str(serialized_path))
    return path if path.is_absolute() else dataset_dir / path


def _load_rasters(path: Path) -> np.ndarray:
    # Place physical channels after time so each item is [T, C, H, W]
    with np.load(path, allow_pickle=False) as bundle:
        return np.stack([bundle[name] for name in MODEL_IMAGE_KEYS], axis=1)


def _load_raster_bundle(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Load physical channels and masks for reconstruction inputs
    with np.load(path, allow_pickle=False) as bundle:
        rasters = np.stack([bundle[name] for name in MODEL_IMAGE_KEYS], axis=1)
        masks = np.stack([bundle[name] for name in MODEL_MASK_KEYS], axis=1).astype(np.float32)
    return rasters, masks


def _safe_scale(values: np.ndarray) -> np.ndarray:
    # Replace unusable normalization scales with one
    return np.where(np.isfinite(values) & (values > MIN_SCALE), values, 1.0)


def _channel_valid_mask(rasters: np.ndarray, channel: int) -> np.ndarray:
    # Identify valid pixels for one numeric image channel
    return np.isfinite(rasters[:, channel])


def _valid_channel_values(rasters: np.ndarray, channel: int) -> np.ndarray:
    # Select valid values for one numeric image channel
    return rasters[:, channel][_channel_valid_mask(rasters, channel)]


def _update_moments(
    values: np.ndarray,
    count: int,
    mean: float,
    sum_squared_deviation: float,
) -> tuple[float, float]:
    # Combine one pixel batch with running population moments accumulated over count pixels
    batch_count = int(values.size)
    if batch_count == 0:
        return mean, sum_squared_deviation
    batch_mean = float(values.mean())
    batch_squared_deviation = float(np.square(values - batch_mean).sum())
    combined_count = count + batch_count
    delta = batch_mean - mean
    combined_mean = mean + delta * batch_count / combined_count
    combined_deviation = (
        sum_squared_deviation + batch_squared_deviation + delta * delta * count * batch_count / combined_count
    )
    return combined_mean, combined_deviation


def _fit_image_stats(
    raster_paths: np.ndarray,
    root: Path,
    progress_interval: int,
) -> tuple[np.ndarray, ...]:
    # Accumulate train-split moments for every numeric raster channel
    channel_count = len(MODEL_IMAGE_KEYS)
    count = np.zeros(channel_count, dtype=np.int64)
    mean = np.zeros(channel_count, dtype=np.float64)
    sum_squared_deviation = np.zeros(channel_count, dtype=np.float64)
    for index, serialized_path in enumerate(raster_paths, start=1):
        rasters = _load_rasters(_raster_path(serialized_path, root))
        for channel in range(channel_count):
            values = _valid_channel_values(rasters, channel)
            mean[channel], sum_squared_deviation[channel] = _update_moments(
                values.astype(np.float64, copy=False),
                int(count[channel]),
                float(mean[channel]),
                float(sum_squared_deviation[channel]),
            )
            count[channel] += values.size
        if progress_interval > 0 and (index % progress_interval == 0 or index == len(raster_paths)):
            print(f"Image-statistics scan: {index:,}/{len(raster_paths):,} rasters")

    center = mean.copy()
    scale = np.ones(channel_count, dtype=np.float64)
    for channel in range(channel_count):
        if count[channel] > 0:
            scale[channel] = np.sqrt(sum_squared_deviation[channel] / count[channel])
    return center, _safe_scale(scale), count


def compute_stats(
    split: str = "train",
    *,
    dataset_dir: str | Path = DATASET_DIR,
    dataframe_dir: str | Path = DATASET_DF,
    progress_interval: int = 1_000,
) -> NormalizationStats:
    """Compute memory-bounded normalization statistics from one split.

    Args:
        split: Dataset split used to estimate statistics.
        dataset_dir: Root containing raster bundles.
        dataframe_dir: Directory containing split CSV files.
        progress_interval: Records between progress messages.

    Returns:
        Frozen image and seasonal normalization statistics.
    """
    root = Path(dataset_dir)
    frame = _read_split_frame(split, Path(dataframe_dir))
    features = _feature_matrix(frame)
    raster_paths = frame[RASTER_PATH_COL].to_numpy(dtype=str)
    image_center, image_scale, image_valid_pixels = _fit_image_stats(raster_paths, root, progress_interval)
    feature_std = _safe_scale(features.std(axis=0))
    return NormalizationStats(
        image_keys=MODEL_IMAGE_KEYS,
        image_center=tuple(float(value) for value in image_center),
        image_scale=tuple(float(value) for value in image_scale),
        image_valid_pixels=tuple(int(value) for value in image_valid_pixels),
        feature_names=MODEL_FEATURE_NAMES,
        feature_mean=tuple(float(value) for value in features.mean(axis=0)),
        feature_std=tuple(float(value) for value in feature_std),
        target_name=MODEL_TARGET_COL,
        training_records=len(frame),
    )


def clipped_pixel_fractions(
    split: str,
    stats: NormalizationStats,
    *,
    dataset_dir: str | Path = DATASET_DIR,
    dataframe_dir: str | Path = DATASET_DF,
) -> dict[str, float]:
    """Calculate the clipped share of valid pixels for one split.

    Args:
        split: Dataset split to inspect.
        stats: Frozen training normalization statistics.
        dataset_dir: Root containing raster bundles.
        dataframe_dir: Directory containing split CSV files.

    Returns:
        Clipped valid-pixel fraction keyed by numeric channel.
    """
    root = Path(dataset_dir)
    frame = _read_split_frame(split, Path(dataframe_dir))
    clipped = np.zeros(len(MODEL_IMAGE_KEYS), dtype=np.int64)
    valid = np.zeros(len(MODEL_IMAGE_KEYS), dtype=np.int64)
    center = np.asarray(stats.image_center)
    scale = np.asarray(stats.image_scale)
    for serialized_path in frame[RASTER_PATH_COL]:
        rasters = _load_rasters(_raster_path(serialized_path, root))
        for channel in range(len(MODEL_IMAGE_KEYS)):
            values = _valid_channel_values(rasters, channel)
            normalized = (values - center[channel]) / scale[channel]
            clipped[channel] += np.count_nonzero(np.abs(normalized) > MODEL_IMAGE_CLIP_ABS)
            valid[channel] += values.size
    return {
        name: float(channel_clipped / channel_valid) if channel_valid else 0.0
        for name, channel_clipped, channel_valid in zip(MODEL_IMAGE_KEYS, clipped, valid, strict=True)
    }


def save_stats(stats: NormalizationStats, path: str | Path) -> None:
    """Atomically persist preprocessing state beside a model checkpoint.

    Args:
        stats: Normalization state to save.
        path: Destination JSON path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(stats.to_dict(), temporary, indent=2)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


class NOxDataset(Dataset):
    """Lazy per-record TEMPO raster and seasonal-feature dataset."""

    def __init__(
        self,
        split: str,
        stats: NormalizationStats | dict[str, object],
        *,
        dataset_dir: str | Path = DATASET_DIR,
        dataframe_dir: str | Path = DATASET_DF,
        load_images: bool = True,
        completed_raster_path: str | Path | None = None,
        source_image_stats: dict[str, object] | None = None,
        transport_transform: TransportTransform | None = None,
        contribution_target_path: str | Path | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.dataframe_dir = Path(dataframe_dir)
        self.load_images = load_images
        self.completed_raster_path = Path(completed_raster_path) if completed_raster_path else None
        self.transport_transform = transport_transform
        self._completed_rasters: np.ndarray | None = None
        self.contribution_target_path = Path(contribution_target_path) if contribution_target_path else None
        self._contribution_targets: np.ndarray | None = None
        self.frame = _read_split_frame(split, self.dataframe_dir)
        self.stats = stats if isinstance(stats, NormalizationStats) else NormalizationStats.from_dict(stats)
        self.image_center = np.asarray(self.stats.image_center, dtype=np.float32)
        self.image_scale = np.asarray(self.stats.image_scale, dtype=np.float32)
        source_stats = source_image_stats or self.stats.to_dict()
        self.source_image_center = np.asarray(source_stats["image_center"], dtype=np.float32)
        self.source_image_scale = np.asarray(source_stats["image_scale"], dtype=np.float32)
        self.no2_center_scale = np.broadcast_to(
            np.asarray(
                (self.image_center[NO2_CHANNEL], self.image_scale[NO2_CHANNEL]),
                dtype=np.float32,
            ),
            (len(self.frame), 2),
        ).copy()

        raw_features = _feature_matrix(self.frame)
        feature_mean = np.asarray(self.stats.feature_mean, dtype=np.float64)
        feature_std = np.asarray(self.stats.feature_std, dtype=np.float64)
        self.features = ((raw_features - feature_mean) / feature_std).astype(np.float32)
        class_indices = {name: index for index, name in enumerate(MODEL_CLASS_NAMES)}
        labels = self.frame[MODEL_TARGET_COL].map(class_indices)
        if labels.isna().any():
            unknown = sorted(self.frame.loc[labels.isna(), MODEL_TARGET_COL].astype(str).unique())
            raise ValueError(f"Unsupported {MODEL_TARGET_COL} values: {', '.join(unknown)}")
        self.labels = labels.to_numpy(dtype=np.int64)
        self.raster_paths = self.frame[RASTER_PATH_COL].to_numpy(dtype=str)
        timestep_times = np.column_stack(
            [
                pd.to_datetime(self.frame[column], utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
                for column in TIMESTEP_TIME_COLUMNS
            ]
        )
        self.elapsed_hours = (np.diff(timestep_times, axis=1) / np.timedelta64(1, "h")).astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def _normalize_completed(self, completed: np.ndarray) -> torch.Tensor:
        # Convert masked-model normalization into classifier normalization
        numeric = completed[:, : len(MODEL_IMAGE_KEYS)]
        source_center = self.source_image_center[None, :, None, None]
        source_scale = self.source_image_scale[None, :, None, None]
        physical = numeric * source_scale + source_center
        center = self.image_center[None, :, None, None]
        scale = self.image_scale[None, :, None, None]
        numeric[:] = (physical - center) / scale
        np.clip(completed, -MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_CLIP_ABS, out=completed)
        return torch.from_numpy(completed)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if self.load_images:
            if self.completed_raster_path is not None:
                if self._completed_rasters is None:
                    self._completed_rasters = np.load(self.completed_raster_path, mmap_mode="r")
                completed = np.array(self._completed_rasters[index], copy=True)
                image = self._normalize_completed(completed)
            else:
                raster_path = _raster_path(self.raster_paths[index], self.dataset_dir)
                rasters, masks = _load_raster_bundle(raster_path)
                normalized = np.zeros_like(rasters, dtype=np.float32)
                for channel in range(len(MODEL_IMAGE_KEYS)):
                    channel_valid = _channel_valid_mask(rasters, channel)
                    channel_values = rasters[:, channel][channel_valid]
                    normalized_channel = normalized[:, channel]
                    normalized_channel[channel_valid] = (
                        channel_values - self.source_image_center[channel]
                    ) / self.source_image_scale[channel]
                np.clip(normalized, -MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_CLIP_ABS, out=normalized)
                image = torch.from_numpy(np.concatenate((normalized, masks), axis=1))
        else:
            image = torch.empty(0, dtype=torch.float32)
        transport = torch.empty(0, dtype=torch.float32)
        if self.transport_transform is not None:
            elapsed_hours = torch.from_numpy(self.elapsed_hours[index]).unsqueeze(0)
            no2_center_scale = torch.from_numpy(self.no2_center_scale[index]).unsqueeze(0)
            residual, quality = self.transport_transform(
                image.unsqueeze(0),
                elapsed_hours,
                no2_center_scale,
            )
            transport = torch.cat((residual, quality), dim=2).squeeze(0)
        item = (
            image,
            torch.from_numpy(self.features[index]),
            transport,
            torch.tensor(self.labels[index]),
            index,
        )
        if self.contribution_target_path is None:
            return item
        if self._contribution_targets is None:
            self._contribution_targets = np.load(self.contribution_target_path, mmap_mode="r")
        target = np.array(self._contribution_targets[index], dtype=np.float32, copy=True)
        return (*item, torch.from_numpy(target))
