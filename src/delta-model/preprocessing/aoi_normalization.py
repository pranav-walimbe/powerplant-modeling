"""Compute robust AOI-specific normalization statistics for TEMPO NO2 rasters."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from preprocessing.generate_dataset_utils import (
    NO2_RASTER_NAME,
    ScanTask,
    bounded_parallel_map,
    make_scan_task,
    process_scan,
    write_json_atomic,
)
from preprocessing.stratify_utils import AOI_ID_COL

from config import (
    DATASET_AOI_NORMALIZATION_STATS,
    DATASET_DF,
    DATASET_TEMPO_CACHE_DIR,
    DATASET_WORKERS_PER_SHARD,
    TEMPO_AOI_MAPPING,
    TEMPO_DIR,
    VIS_DIR,
)

DEFAULT_OUTPUT = Path(DATASET_AOI_NORMALIZATION_STATS)
DEFAULT_PLOT_OUTPUT = Path(VIS_DIR) / "aoi-no2-normalization.png"
DEFAULT_RASTERS_PER_MONTH = 60
DEFAULT_RANDOM_SEED = 42
MONTHS = tuple(range(1, 13))
MIN_SCALE = 1e-12
ROBUST_STD_NORMALIZER = 1.349
SPLIT_PATHS = tuple(Path(DATASET_DF) / f"{split}_df.csv" for split in ("train", "val", "test"))


@dataclass(frozen=True)
class AoiNormalizationTask:
    """Selected TEMPO scans used to fit one AOI's normalization."""

    aoi_id: int
    scans: tuple[ScanTask, ...]


@dataclass(frozen=True)
class AoiNormalizationResult:
    """Robust NO2 normalization fitted for one AOI."""

    aoi_id: int
    median: float
    scale: float
    valid_pixels: int
    rasters: int


def _positive_int(value: str) -> int:
    # Parse a strictly positive command-line integer
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse AOI normalization command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT_OUTPUT)
    parser.add_argument("--rasters-per-month", type=_positive_int, default=DEFAULT_RASTERS_PER_MONTH)
    parser.add_argument("--workers", type=_positive_int, default=DATASET_WORKERS_PER_SHARD)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def _load_target_aois() -> pl.DataFrame:
    # Collect one stable coordinate pair for every AOI in the finalized dataset
    frames = []
    for path in SPLIT_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"Finalized dataset output is missing: {path}")
        frames.append(pl.scan_csv(path).select(AOI_ID_COL, "lat", "lon").unique().collect())
    aois = pl.concat(frames).unique().sort(AOI_ID_COL)
    coordinate_counts = aois.group_by(AOI_ID_COL).len()
    inconsistent = coordinate_counts.filter(pl.col("len") != 1)
    if not inconsistent.is_empty():
        identifiers = ", ".join(str(value) for value in inconsistent[AOI_ID_COL].to_list())
        raise ValueError(f"AOIs have inconsistent coordinates across stratification outputs: {identifiers}")
    return aois


def _scan_task(row: dict[str, object], aoi: dict[str, object]) -> ScanTask:
    # Reuse the dataset generator's exact cache identity
    return make_scan_task(
        {
            AOI_ID_COL: int(aoi[AOI_ID_COL]),
            "lat": float(aoi["lat"]),
            "lon": float(aoi["lon"]),
            "granule_paths": row["granule_paths"],
        },
        "granule_paths",
        Path(TEMPO_DIR),
        Path(DATASET_TEMPO_CACHE_DIR),
    )


def _select_month_scans(
    candidates: pl.DataFrame,
    aoi: dict[str, object],
    month: int,
    quota: int,
    seed: int,
) -> tuple[tuple[ScanTask, ...], int]:
    # Probe randomized candidates until the cache supplies the quota or the month is exhausted
    random = np.random.default_rng(np.random.SeedSequence((seed, int(aoi[AOI_ID_COL]), month)))
    order = random.permutation(candidates.height)
    hits: list[ScanTask] = []
    misses: list[ScanTask] = []
    seen_keys: set[str] = set()
    rows = candidates.to_dicts()
    for index in order:
        task = _scan_task(rows[int(index)], aoi)
        if task.cache_key in seen_keys:
            continue
        seen_keys.add(task.cache_key)
        if Path(task.cache_path).is_file():
            hits.append(task)
            if len(hits) == quota:
                return tuple(hits), len(hits)
        else:
            misses.append(task)

    selected_misses = misses[: max(quota - len(hits), 0)]
    return tuple((*hits, *selected_misses)), len(hits)


def _build_aoi_task(
    aoi: dict[str, object],
    observations: pl.DataFrame,
    rasters_per_month: int,
    seed: int,
) -> tuple[AoiNormalizationTask, int]:
    # Select cache hits first within each calendar month
    selected: list[ScanTask] = []
    cache_hits = 0
    for month in MONTHS:
        candidates = observations.filter(pl.col("month") == month)
        month_scans, month_hits = _select_month_scans(candidates, aoi, month, rasters_per_month, seed)
        selected.extend(month_scans)
        cache_hits += month_hits
        if len(month_scans) < rasters_per_month:
            print(
                f"AOI {aoi[AOI_ID_COL]} month {month:02d}: selected "
                f"{len(month_scans):,}/{rasters_per_month:,} available rasters"
            )
    return AoiNormalizationTask(int(aoi[AOI_ID_COL]), tuple(selected)), cache_hits


def _build_tasks(
    aois: pl.DataFrame,
    rasters_per_month: int,
    seed: int,
) -> list[AoiNormalizationTask]:
    # Stream only active AOI rows from the general observation mapping
    mapping_paths = sorted(Path(TEMPO_AOI_MAPPING).rglob("date=*.parquet"))
    if not mapping_paths:
        raise FileNotFoundError(f"TEMPO AOI-observation mapping is empty: {TEMPO_AOI_MAPPING}")
    observations = (
        pl.scan_parquet(mapping_paths)
        .select(AOI_ID_COL, "tempo_time", "granule_paths")
        .join(aois.select(AOI_ID_COL).lazy(), on=AOI_ID_COL, how="semi")
        .with_columns(pl.col("tempo_time").dt.month().alias("month"))
        .collect(engine="streaming")
    )
    empty_observations = observations.clear()
    observations_by_aoi = {
        int(key[0]): frame for key, frame in observations.partition_by(AOI_ID_COL, as_dict=True).items()
    }
    del observations
    tasks = []
    for aoi in aois.iter_rows(named=True):
        aoi_id = int(aoi[AOI_ID_COL])
        aoi_observations = observations_by_aoi.get(
            aoi_id,
            empty_observations,
        )
        task, cache_hits = _build_aoi_task(aoi, aoi_observations, rasters_per_month, seed)
        cache_misses = len(task.scans) - cache_hits
        hit_percentage = 100.0 * cache_hits / len(task.scans) if task.scans else 0.0
        miss_percentage = 100.0 * cache_misses / len(task.scans) if task.scans else 0.0
        print(
            f"AOI {aoi_id}: {len(task.scans):,} samples; {cache_hits:,} cache hits "
            f"({hit_percentage:.1f}%); {cache_misses:,} cache misses ({miss_percentage:.1f}%)"
        )
        tasks.append(task)
    return tasks


def _color_limits(values: np.ndarray) -> tuple[float, float]:
    # Limit map colors to the central range while retaining every AOI marker
    lower, upper = np.percentile(values, (2, 98))
    if np.isclose(lower, upper):
        padding = max(abs(float(lower)) * 0.01, 1.0)
        return float(lower - padding), float(upper + padding)
    return float(lower), float(upper)


def _plot_normalization(
    results: dict[int, AoiNormalizationResult],
    aois: pl.DataFrame,
    destination: Path,
) -> None:
    # Show the geographic and joint variation of fitted AOI statistics
    rows = []
    coordinates = {int(row[AOI_ID_COL]): row for row in aois.iter_rows(named=True)}
    for aoi_id, result in sorted(results.items()):
        coordinate = coordinates[aoi_id]
        rows.append(
            (
                float(coordinate["lon"]),
                float(coordinate["lat"]),
                result.median,
                result.scale,
                result.rasters,
            )
        )
    values = np.asarray(rows, dtype=np.float64)
    longitudes, latitudes, medians, scales, raster_counts = values.T
    median_limits = _color_limits(medians)
    scale_limits = _color_limits(scales)

    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    median_map = axes[0].scatter(
        longitudes,
        latitudes,
        c=medians,
        cmap="viridis",
        s=24,
        vmin=median_limits[0],
        vmax=median_limits[1],
    )
    figure.colorbar(median_map, ax=axes[0], label="NO2 median")
    axes[0].set(title="AOI normalization center", xlabel="Longitude", ylabel="Latitude")

    scale_map = axes[1].scatter(
        longitudes,
        latitudes,
        c=scales,
        cmap="magma",
        s=24,
        vmin=scale_limits[0],
        vmax=scale_limits[1],
    )
    figure.colorbar(scale_map, ax=axes[1], label="NO2 robust scale")
    axes[1].set(title="AOI normalization scale", xlabel="Longitude", ylabel="Latitude")

    relationship = axes[2].scatter(medians, scales, c=raster_counts, cmap="plasma", s=24)
    figure.colorbar(relationship, ax=axes[2], label="Rasters sampled")
    axes[2].set(title="Center-scale relationship", xlabel="NO2 median", ylabel="NO2 robust scale")
    figure.suptitle(f"AOI-specific TEMPO NO2 normalization ({len(results):,} AOIs)", fontsize=16)

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def _compute_aoi_normalization(task: AoiNormalizationTask) -> AoiNormalizationResult:
    # Read or generate every selected raster before pooling its finite NO2 pixels
    valid_values: list[np.ndarray] = []
    for scan in task.scans:
        if not Path(scan.cache_path).is_file():
            result = process_scan(scan)
            if result.error is not None:
                raise RuntimeError(f"AOI {task.aoi_id} scan {scan.cache_key} failed: {result.error}")
        with np.load(scan.cache_path, allow_pickle=False) as raster:
            no2 = np.asarray(raster[NO2_RASTER_NAME], dtype=np.float32)
        finite = no2[np.isfinite(no2)]
        if finite.size:
            valid_values.append(finite)

    if not valid_values:
        raise ValueError(f"AOI {task.aoi_id} has no finite NO2 pixels across {len(task.scans):,} rasters")
    pooled = np.concatenate(valid_values)
    lower, median, upper = np.percentile(pooled, (25, 50, 75), overwrite_input=True)
    scale = float((upper - lower) / ROBUST_STD_NORMALIZER)
    if not np.isfinite(scale) or scale <= MIN_SCALE:
        scale = 1.0
    return AoiNormalizationResult(
        aoi_id=task.aoi_id,
        median=float(median),
        scale=scale,
        valid_pixels=int(pooled.size),
        rasters=len(task.scans),
    )


def compute_aoi_no2_normalization(
    output: Path,
    plot_output: Path,
    rasters_per_month: int = DEFAULT_RASTERS_PER_MONTH,
    workers: int = DATASET_WORKERS_PER_SHARD,
    seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """Compute and save robust NO2 normalization for every stratified AOI.

    Args:
        output: Destination JSON mapping AOI identifiers to statistics.
        plot_output: Destination PNG summarizing AOI statistics.
        rasters_per_month: Maximum selected raster count for each AOI-month.
        workers: AOIs processed concurrently.
        seed: Deterministic sampling seed.
    """
    aois = _load_target_aois()
    tasks = _build_tasks(aois, rasters_per_month, seed)
    results: dict[int, AoiNormalizationResult] = {}
    for result in bounded_parallel_map(_compute_aoi_normalization, tasks, workers):
        results[result.aoi_id] = result
        print(f"Completed AOI {result.aoi_id}: {len(results):,}/{len(tasks):,}")

    serialized = {
        str(aoi_id): {
            "median": result.median,
            "scale": result.scale,
            "valid_pixels": result.valid_pixels,
            "rasters": result.rasters,
        }
        for aoi_id, result in sorted(results.items())
    }
    write_json_atomic(serialized, output)
    _plot_normalization(results, aois, plot_output)
    print(f"Saved AOI NO2 normalization for {len(serialized):,} AOIs to {output}")
    print(f"Saved AOI NO2 normalization plot to {plot_output}")


def main() -> None:
    """Run AOI-specific TEMPO NO2 normalization."""
    args = parse_args()
    compute_aoi_no2_normalization(
        args.output,
        args.plot_output,
        args.rasters_per_month,
        args.workers,
        args.seed,
    )


if __name__ == "__main__":
    main()
