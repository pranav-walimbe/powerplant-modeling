"""Dataset loading and train-only normalization for masked NO2 modeling."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    MASKED_PRETRAINING_BASE_DIR,
    MASKED_PRETRAINING_DF_DIR,
    MODEL_IMAGE_CLIP_ABS,
    MODEL_IMAGE_KEYS,
)

RASTER_PATH_COL = "raster_bundle_path"
CACHE_KEY_COL = "cache_key"
MASKED_IMAGE_KEYS = MODEL_IMAGE_KEYS
ORIGINAL_MASK_KEY = "no2_mask"
MIN_SCALE = 1e-12
NORMALIZATION_METHOD = "zscore"
MIN_MASK_FRACTION = 0.01
MAX_MASK_FRACTION = 0.15
FRONTIER_SELECTION_PROBABILITY = 0.94
EDGE_DECAY_PIXELS = 2.5
EDGE_WEIGHT_FLOOR = 0.06


@dataclass(frozen=True)
class MaskedNormalizationStats:
    """JSON-safe image normalization fitted on the pretraining train split."""

    image_keys: tuple[str, ...]
    image_center: tuple[float, ...]
    image_scale: tuple[float, ...]
    image_valid_pixels: tuple[int, ...]
    training_records: int
    normalization_method: str = NORMALIZATION_METHOD

    def to_dict(self) -> dict[str, object]:
        """Return normalization state as JSON-safe values."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "MaskedNormalizationStats":
        """Build normalization state from serialized values.

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
            training_records=int(values["training_records"]),
            normalization_method=str(values["normalization_method"]),
        )


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


def compute_stats(
    *,
    dataset_dir: str | Path = MASKED_PRETRAINING_BASE_DIR,
    dataframe_dir: str | Path = MASKED_PRETRAINING_DF_DIR,
    progress_interval: int = 1_000,
) -> MaskedNormalizationStats:
    """Compute image normalization statistics from the training split.

    Args:
        dataset_dir: Root used to resolve relative raster paths.
        dataframe_dir: Directory containing split CSV files.
        progress_interval: Records between progress messages.

    Returns:
        Frozen train-split normalization statistics.
    """
    frame = pd.read_csv(Path(dataframe_dir) / "train_df.csv")
    paths = frame[RASTER_PATH_COL].to_numpy(dtype=str)
    root = Path(dataset_dir)
    channel_count = len(MASKED_IMAGE_KEYS)
    counts = np.zeros(channel_count, dtype=np.int64)
    mean = np.zeros(channel_count, dtype=np.float64)
    sum_squared_deviation = np.zeros(channel_count, dtype=np.float64)

    for index, serialized_path in enumerate(paths, start=1):
        path = Path(serialized_path)
        path = path if path.is_absolute() else root / path
        with np.load(path, allow_pickle=False) as bundle:
            for channel, name in enumerate(MASKED_IMAGE_KEYS):
                values = np.asarray(bundle[name], dtype=np.float64)
                valid = values[np.isfinite(values)]
                mean[channel], sum_squared_deviation[channel] = _update_moments(
                    valid,
                    int(counts[channel]),
                    float(mean[channel]),
                    float(sum_squared_deviation[channel]),
                )
                counts[channel] += valid.size
        if progress_interval > 0 and (index % progress_interval == 0 or index == len(paths)):
            print(f"Image-statistics scan: {index:,}/{len(paths):,} rasters")

    center = mean.copy()
    scale = np.sqrt(
        np.divide(
            sum_squared_deviation,
            counts,
            out=np.zeros_like(sum_squared_deviation),
            where=counts > 0,
        )
    )
    scale = np.where(np.isfinite(scale) & (scale > MIN_SCALE), scale, 1.0)
    return MaskedNormalizationStats(
        image_keys=MASKED_IMAGE_KEYS,
        image_center=tuple(float(value) for value in center),
        image_scale=tuple(float(value) for value in scale),
        image_valid_pixels=tuple(int(value) for value in counts),
        training_records=len(frame),
    )


def save_stats(stats: MaskedNormalizationStats, path: str | Path) -> None:
    """Atomically persist preprocessing state beside the model checkpoint.

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
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _record_rng(identifier: str, seed: int) -> np.random.Generator:
    # Derive a platform-stable random stream from the record and run seed
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], byteorder="little"))


def _edge_weights(shape: tuple[int, int]) -> np.ndarray:
    # Assign the highest sampling weight at the raster boundary
    rows, columns = np.indices(shape)
    edge_distance = np.minimum.reduce((rows, columns, shape[0] - 1 - rows, shape[1] - 1 - columns))
    return np.exp(-edge_distance / EDGE_DECAY_PIXELS).ravel() + EDGE_WEIGHT_FLOOR


def _weighted_choice(
    candidates: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> int:
    # Draw one candidate after normalizing its positive weights
    candidate_weights = weights[candidates]
    probabilities = candidate_weights / candidate_weights.sum()
    return int(candidates[rng.choice(len(candidates), p=probabilities)])


def _neighbors(pixel: int, shape: tuple[int, int]) -> tuple[int, ...]:
    # Return valid eight-connected neighbors in stable order
    row, column = divmod(pixel, shape[1])
    return tuple(
        neighbor_row * shape[1] + neighbor_column
        for neighbor_row in range(max(0, row - 1), min(shape[0], row + 2))
        for neighbor_column in range(max(0, column - 1), min(shape[1], column + 2))
        if neighbor_row != row or neighbor_column != column
    )


def _synthetic_visible_mask(
    shape: tuple[int, int],
    identifier: str,
    seed: int,
) -> np.ndarray:
    # Grow deterministic edge-biased missing regions from the record hash
    rng = _record_rng(identifier, seed)
    missing_fraction = rng.uniform(MIN_MASK_FRACTION, MAX_MASK_FRACTION)
    target_count = max(1, round(missing_fraction * shape[0] * shape[1]))
    edge_weights = _edge_weights(shape)
    missing = np.zeros(shape[0] * shape[1], dtype=bool)
    frontier: dict[int, int] = {}
    missing_count = 0

    while missing_count < target_count:
        if frontier and rng.random() < FRONTIER_SELECTION_PROBABILITY:
            candidates = np.fromiter(frontier, dtype=np.int64)
            neighbor_counts = np.fromiter(frontier.values(), dtype=np.float64)
            weights = edge_weights.copy()
            weights[candidates] *= np.square(1.0 + neighbor_counts)
        else:
            candidates = np.flatnonzero(~missing)
            weights = edge_weights

        selected = _weighted_choice(candidates, weights, rng)
        missing[selected] = True
        missing_count += 1
        frontier.pop(selected, None)
        for neighbor in _neighbors(selected, shape):
            if not missing[neighbor]:
                frontier[neighbor] = frontier.get(neighbor, 0) + 1

    return (~missing).reshape(shape)


class MaskedNO2Dataset(Dataset):
    """Lazy masked-raster reconstruction dataset."""

    def __init__(
        self,
        split: str,
        stats: MaskedNormalizationStats | dict[str, object],
        *,
        dataset_dir: str | Path = MASKED_PRETRAINING_BASE_DIR,
        dataframe_dir: str | Path = MASKED_PRETRAINING_DF_DIR,
        seed: int = 42,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.frame = pd.read_csv(Path(dataframe_dir) / f"{split}_df.csv")
        self.stats = stats if isinstance(stats, MaskedNormalizationStats) else MaskedNormalizationStats.from_dict(stats)
        self.raster_paths = self.frame[RASTER_PATH_COL].to_numpy(dtype=str)
        self.cache_keys = self.frame[CACHE_KEY_COL].to_numpy(dtype=str)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.raster_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        path = Path(self.raster_paths[index])
        path = path if path.is_absolute() else self.dataset_dir / path
        with np.load(path, allow_pickle=False) as bundle:
            rasters = np.stack([np.asarray(bundle[name], dtype=np.float32) for name in MASKED_IMAGE_KEYS])
            original_valid = np.asarray(bundle[ORIGINAL_MASK_KEY], dtype=bool)

        artificial_visible = _synthetic_visible_mask(original_valid.shape, self.cache_keys[index], self.seed)

        center = np.asarray(self.stats.image_center, dtype=np.float32)[:, None, None]
        scale = np.asarray(self.stats.image_scale, dtype=np.float32)[:, None, None]
        normalized = (rasters - center) / scale
        np.clip(normalized, -MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_CLIP_ABS, out=normalized)
        visible = original_valid & artificial_visible
        target = normalized[:1].copy()
        normalized[0] *= visible
        image = np.concatenate((normalized, visible[None].astype(np.float32)), axis=0)
        loss_mask = (original_valid & ~artificial_visible)[None].astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(target), torch.from_numpy(loss_mask), index
