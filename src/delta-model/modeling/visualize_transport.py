"""Visualize baseline transport predictions against observed NO2 rasters."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from modeling.transport import AdvectionTransform

from config import (
    DATASET_DF,
    DATASET_DIR,
    IMG_SIZE,
    MODEL_CLASS_NAMES,
    MODEL_IMAGE_KEYS,
    MODEL_MASK_KEYS,
    MODEL_TARGET_COL,
    SEQUENCE_TIMESTEPS,
    VIS_DIR,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RASTER_PATH_COL = "raster_bundle_path"
AOI_COL = "aoi_id"
NO2_KEY = "no2"
NO2_MASK_KEY = "no2_mask"
WIND_U_KEY = "wind_u_80m_mps"
WIND_V_KEY = "wind_v_80m_mps"
TIMESTAMP_COLUMNS = tuple(f"t{index}_timestamp" for index in range(SEQUENCE_TIMESTEPS))
SECONDS_PER_HOUR = 3_600.0
NO2_DISPLAY_SCALE = 1e15
DISPLAY_QUANTILES = (0.01, 0.99)
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_SEED = 42
DEFAULT_TRANSITION = SEQUENCE_TIMESTEPS - 2
FIGURE_DPI = 150


@dataclass(frozen=True)
class TransportSample:
    """One observed transition and its horizontal-advection prediction."""

    record_index: int
    aoi_id: int
    target: str
    source_time: str
    current_time: str
    elapsed_hours: float
    mean_wind_u_mps: float
    mean_wind_v_mps: float
    weighted_mae: float
    source_observed: np.ndarray
    predicted: np.ndarray
    current_observed: np.ndarray
    residual: np.ndarray
    residual_quality: np.ndarray


def _parse_args() -> argparse.Namespace:
    # Parse deterministic sampling and output options
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--transition", type=int, choices=range(SEQUENCE_TIMESTEPS - 1), default=DEFAULT_TRANSITION)
    parser.add_argument("--dataset-dir", type=Path, default=Path(DATASET_DIR))
    parser.add_argument("--dataframe-dir", type=Path, default=Path(DATASET_DF))
    run_id = os.getenv("SLURM_JOB_ID", "latest")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(VIS_DIR) / f"transport-predictions-{run_id}" / "transport_predictions.png",
    )
    return parser.parse_args()


def _aoi_diverse_indices(
    frame: pd.DataFrame,
    count: int,
    generator: np.random.Generator,
) -> list[int]:
    # Prefer distinct AOIs within each target class
    shuffled = generator.permutation(frame.index.to_numpy())
    distinct: list[int] = []
    repeated: list[int] = []
    seen_aois: set[int] = set()
    for index in shuffled:
        aoi_id = int(frame.at[index, AOI_COL])
        destination = distinct if aoi_id not in seen_aois else repeated
        destination.append(int(index))
        seen_aois.add(aoi_id)
    return (distinct + repeated)[:count]


def _sample_manifest(
    dataframe_path: Path,
    sample_count: int,
    seed: int,
) -> pd.DataFrame:
    # Select a deterministic class-balanced sample
    if sample_count < len(MODEL_CLASS_NAMES):
        raise ValueError(f"samples must be at least {len(MODEL_CLASS_NAMES)}")
    required_columns = [AOI_COL, MODEL_TARGET_COL, RASTER_PATH_COL, *TIMESTAMP_COLUMNS]
    frame = pd.read_csv(dataframe_path, usecols=required_columns)
    generator = np.random.default_rng(seed)
    base_count, remainder = divmod(sample_count, len(MODEL_CLASS_NAMES))
    selected_indices = []
    for class_index, class_name in enumerate(MODEL_CLASS_NAMES):
        class_count = base_count + int(class_index < remainder)
        class_frame = frame.loc[frame[MODEL_TARGET_COL] == class_name]
        if len(class_frame) < class_count:
            raise ValueError(f"Class {class_name} has only {len(class_frame):,} records")
        selected_indices.extend(_aoi_diverse_indices(class_frame, class_count, generator))
    manifest = frame.loc[selected_indices].copy()
    manifest.insert(0, "record_index", selected_indices)
    return manifest.reset_index(drop=True)


def _fill_missing_bilinear(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # Fill gaps along both raster axes while preserving observations
    if values.shape != (IMG_SIZE, IMG_SIZE) or valid.shape != values.shape:
        raise ValueError("NO2 raster and mask must match the configured image shape")
    if not np.any(valid):
        raise ValueError("Cannot interpolate an empty NO2 raster")
    filled = np.where(valid, values, np.nan).astype(np.float64, copy=True)
    positions = np.arange(IMG_SIZE)
    for row in range(IMG_SIZE):
        finite = np.isfinite(filled[row])
        if np.any(finite):
            filled[row] = np.interp(positions, positions[finite], filled[row, finite])
    for column in range(IMG_SIZE):
        finite = np.isfinite(filled[:, column])
        if np.any(finite):
            filled[:, column] = np.interp(positions, positions[finite], filled[finite, column])
    if not np.isfinite(filled).all():
        filled = np.nan_to_num(filled, nan=float(np.nanmedian(values[valid])))
    filled[valid] = values[valid]
    return filled.astype(np.float32)


def _resolve_raster_path(serialized_path: object, dataset_dir: Path) -> Path:
    # Resolve a stored bundle path against the dataset root
    path = Path(str(serialized_path))
    return path if path.is_absolute() else dataset_dir / path


def _elapsed_hours(row: pd.Series) -> np.ndarray:
    # Calculate the four observed scan intervals
    timestamps = pd.to_datetime([row[column] for column in TIMESTAMP_COLUMNS], utc=True)
    elapsed = np.diff(timestamps.asi8).astype(np.float64) / (SECONDS_PER_HOUR * 1e9)
    if elapsed.shape != (SEQUENCE_TIMESTEPS - 1,) or np.any(~np.isfinite(elapsed)) or np.any(elapsed <= 0):
        raise ValueError(f"Invalid observation intervals for record {row['record_index']}")
    return elapsed.astype(np.float32)


def _load_transport_sample(
    row: pd.Series,
    dataset_dir: Path,
    transition: int,
) -> TransportSample:
    # Load physical rasters and run horizontal advection only
    raster_path = _resolve_raster_path(row[RASTER_PATH_COL], dataset_dir)
    with np.load(raster_path, allow_pickle=False) as bundle:
        missing = set((*MODEL_IMAGE_KEYS, *MODEL_MASK_KEYS)) - set(bundle.files)
        if missing:
            raise ValueError(f"Raster bundle {raster_path} is missing arrays: {sorted(missing)}")
        rasters = {key: np.asarray(bundle[key], dtype=np.float32) for key in MODEL_IMAGE_KEYS}
        no2_quality = np.asarray(bundle[NO2_MASK_KEY], dtype=np.float32)

    filled_no2 = np.stack(
        [
            _fill_missing_bilinear(rasters[NO2_KEY][timestep], no2_quality[timestep].astype(bool))
            for timestep in range(SEQUENCE_TIMESTEPS)
        ]
    )
    image_channels = [filled_no2 if key == NO2_KEY else rasters[key] for key in MODEL_IMAGE_KEYS]
    image_channels.append(no2_quality)
    image = torch.from_numpy(np.stack(image_channels, axis=1)).unsqueeze(0)
    elapsed = _elapsed_hours(row)
    identity_center = tuple(0.0 for _ in MODEL_IMAGE_KEYS)
    identity_scale = tuple(1.0 for _ in MODEL_IMAGE_KEYS)
    transport = AdvectionTransform(identity_center, identity_scale)
    residual, residual_quality = transport.physical_residual(
        image,
        torch.from_numpy(elapsed).unsqueeze(0),
        torch.tensor([[0.0, 1.0]]),
    )

    current = filled_no2[transition + 1].astype(np.float64)
    residual_array = residual[0, transition, 0].numpy().astype(np.float64)
    predicted = current - residual_array
    quality = residual_quality[0, transition, 0].numpy().astype(np.float64)
    source_observed = np.where(no2_quality[transition] > 0, filled_no2[transition], np.nan)
    current_observed = np.where(no2_quality[transition + 1] > 0, current, np.nan)
    quality_sum = float(quality.sum())
    weighted_mae = float((quality * np.abs(residual_array)).sum() / quality_sum) if quality_sum else float("nan")
    wind_u = 0.5 * (rasters[WIND_U_KEY][transition] + rasters[WIND_U_KEY][transition + 1])
    wind_v = 0.5 * (rasters[WIND_V_KEY][transition] + rasters[WIND_V_KEY][transition + 1])
    return TransportSample(
        record_index=int(row["record_index"]),
        aoi_id=int(row[AOI_COL]),
        target=str(row[MODEL_TARGET_COL]),
        source_time=str(row[TIMESTAMP_COLUMNS[transition]]),
        current_time=str(row[TIMESTAMP_COLUMNS[transition + 1]]),
        elapsed_hours=float(elapsed[transition]),
        mean_wind_u_mps=float(np.nanmean(wind_u)),
        mean_wind_v_mps=float(np.nanmean(wind_v)),
        weighted_mae=weighted_mae,
        source_observed=source_observed.astype(np.float64),
        predicted=predicted,
        current_observed=current_observed,
        residual=residual_array,
        residual_quality=quality,
    )


def _display_limits(samples: list[TransportSample]) -> tuple[float, float, float]:
    # Fit robust shared scales across the selected samples
    no2_values = np.concatenate(
        [
            raster[np.isfinite(raster)] / NO2_DISPLAY_SCALE
            for sample in samples
            for raster in (sample.source_observed, sample.predicted, sample.current_observed)
        ]
    )
    residual_values = np.concatenate([sample.residual.ravel() / NO2_DISPLAY_SCALE for sample in samples])
    no2_low, no2_high = np.quantile(no2_values, DISPLAY_QUANTILES)
    residual_limit = float(np.quantile(np.abs(residual_values), DISPLAY_QUANTILES[1]))
    if no2_low == no2_high:
        no2_low -= 0.5
        no2_high += 0.5
    return float(no2_low), float(no2_high), max(residual_limit, np.finfo(float).eps)


def _write_figure(samples: list[TransportSample], output_path: Path, split: str, transition: int) -> None:
    # Render one row per transport prediction
    no2_low, no2_high, residual_limit = _display_limits(samples)
    figure, axes = plt.subplots(len(samples), 5, figsize=(15, 2.35 * len(samples)), squeeze=False)
    no2_cmap = matplotlib.colormaps["viridis"].copy()
    no2_cmap.set_bad("#d9d9d9")
    residual_cmap = matplotlib.colormaps["coolwarm"].copy()
    quality_cmap = matplotlib.colormaps["magma"]
    no2_image = None
    residual_image = None
    quality_image = None
    column_titles = (
        f"Observed source t{transition}",
        "Advected prediction",
        f"Observed current t{transition + 1}",
        "Current - prediction",
        "Residual quality",
    )
    for row_index, sample in enumerate(samples):
        rasters = (
            sample.source_observed / NO2_DISPLAY_SCALE,
            sample.predicted / NO2_DISPLAY_SCALE,
            sample.current_observed / NO2_DISPLAY_SCALE,
            sample.residual / NO2_DISPLAY_SCALE,
            sample.residual_quality,
        )
        for column_index, raster in enumerate(rasters):
            axis = axes[row_index, column_index]
            if column_index < 3:
                no2_image = axis.imshow(raster, cmap=no2_cmap, vmin=no2_low, vmax=no2_high)
            elif column_index == 3:
                residual_image = axis.imshow(
                    raster,
                    cmap=residual_cmap,
                    vmin=-residual_limit,
                    vmax=residual_limit,
                )
            else:
                quality_image = axis.imshow(raster, cmap=quality_cmap, vmin=0.0, vmax=1.0)
            if row_index == 0:
                axis.set_title(column_titles[column_index], fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(
            f"row {sample.record_index} | AOI {sample.aoi_id} | {sample.target}\n"
            f"dt={sample.elapsed_hours:.2f} h | u={sample.mean_wind_u_mps:+.1f}, "
            f"v={sample.mean_wind_v_mps:+.1f} m/s\n"
            f"quality-weighted MAE={sample.weighted_mae / NO2_DISPLAY_SCALE:.2f}",
            fontsize=7,
        )

    figure.suptitle(
        f"Horizontal-advection baseline on {len(samples)} class-balanced {split} samples\n"
        "Missing NO2 filled by spatial interpolation; no decay or elevated-wind adjustment",
        fontsize=12,
    )
    figure.subplots_adjust(left=0.16, right=0.91, top=0.94, bottom=0.05, wspace=0.08, hspace=0.16)
    if no2_image is None or residual_image is None or quality_image is None:
        raise RuntimeError("No transport samples were rendered")
    figure.colorbar(no2_image, ax=axes[:, :3], fraction=0.012, pad=0.015, label="NO2 / 1e15")
    figure.colorbar(residual_image, ax=axes[:, 3], fraction=0.035, pad=0.03, label="Residual / 1e15")
    figure.colorbar(quality_image, ax=axes[:, 4], fraction=0.035, pad=0.03, label="Quality")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def _write_manifest(samples: list[TransportSample], output_path: Path) -> None:
    # Save the selected records and diagnostic scalars
    rows = [
        {
            "record_index": sample.record_index,
            "aoi_id": sample.aoi_id,
            "target": sample.target,
            "source_time": sample.source_time,
            "current_time": sample.current_time,
            "elapsed_hours": sample.elapsed_hours,
            "mean_wind_u_mps": sample.mean_wind_u_mps,
            "mean_wind_v_mps": sample.mean_wind_v_mps,
            "quality_weighted_mae": sample.weighted_mae,
        }
        for sample in samples
    ]
    pd.DataFrame(rows).to_csv(output_path.with_suffix(".csv"), index=False)


def main() -> None:
    """Generate the deterministic transport-prediction montage."""
    args = _parse_args()
    dataframe_path = args.dataframe_dir / f"{args.split}_df.csv"
    manifest = _sample_manifest(dataframe_path, args.samples, args.seed)
    samples = [
        _load_transport_sample(row, args.dataset_dir, args.transition)
        for _, row in manifest.iterrows()
    ]
    _write_figure(samples, args.output, args.split, args.transition)
    _write_manifest(samples, args.output)
    print(f"Saved transport prediction montage to {args.output}")


if __name__ == "__main__":
    main()
