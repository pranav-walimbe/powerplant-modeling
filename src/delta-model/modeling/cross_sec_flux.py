"""Cross-sectional flux estimates for source-centered TEMPO NO2 rasters.

This module adapts the single-image CSF method described by Kuhlmann et al.
(2024), https://doi.org/10.5194/gmd-17-4773-2024, and Meier et al. (2024),
https://doi.org/10.5194/acp-24-7667-2024. The Gaussian transect model and its
bounds follow the open-source ddeq implementation at revision 7a9c46e:
https://gitlab.com/efmkoene/ddeq/-/blob/7a9c46e/src/ddeq/csf.py

The adaptation operates directly on the repository's regular 24 by 24 AOI
grid. It uses a straight centerline aligned with the source wind instead of
ddeq's detected curved plume centerline. This difference must be evaluated
before treating the estimates as a literature-reproduction baseline.
"""

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modeling.dataset import RASTER_PATH_COL, TIMESTEP_TIME_COLUMNS
from scipy.ndimage import map_coordinates
from scipy.optimize import OptimizeWarning, curve_fit

from config import (
    DATASET_DF,
    DATASET_DIR,
    IMG_SIZE,
    MODEL_CLASS_NAMES,
    MODEL_TARGET_COL,
    STRATIFICATION_WINDOW_THRESHOLD,
)

AVOGADRO_MOLECULES_PER_MOL = 6.02214076e23
NO2_KG_PER_MOL = 46.0055e-3
CM2_PER_M2 = 1.0e4
KG_PER_SECOND_TO_LB_PER_HOUR = 2.2046226218 * 3600.0
AOI_CENTRE_PIXEL = (IMG_SIZE - 1) / 2.0
TIMESTEP_NOX_COLUMNS = tuple(f"t{index}_nox" for index in range(5))


@dataclass(frozen=True)
class CrossSectionFluxConfig:
    """Configuration for one source-level CSF estimate."""

    cell_size_m: float = 3_000.0
    maximum_downwind_m: float = 30_000.0
    maximum_crosswind_m: float = 30_000.0
    minimum_wind_speed_mps: float = 2.0
    maximum_wind_speed_mps: float = 10.0
    no2_to_nox_factor: float = 1.32
    nox_lifetime_seconds: float | None = 4.0 * 3600.0
    minimum_transects: int = 3


@dataclass(frozen=True)
class CrossSectionFluxEstimate:
    """Source emission estimate and its fitted transect diagnostics."""

    emission_lb_per_hour: float
    emission_precision_lb_per_hour: float
    wind_speed_mps: float
    downwind_distance_m: np.ndarray
    line_density_kg_per_m: np.ndarray
    flux_kg_per_second: np.ndarray
    valid_transects: int


@dataclass(frozen=True)
class CrossSectionFluxSequenceEstimate:
    """Five-timestep AOI estimate and its label-compatible summaries."""

    timestep_emission_lb_per_hour: np.ndarray
    timestep_precision_lb_per_hour: np.ndarray
    timestep_wind_speed_mps: np.ndarray
    timestep_valid_transects: np.ndarray
    early_emission_lb_per_hour: float
    late_emission_lb_per_hour: float
    delta_emission_lb_per_hour: float
    predicted_class: str | None


def tempo_no2_to_mass_column(no2_molecules_per_cm2: np.ndarray) -> np.ndarray:
    """Convert TEMPO NO2 vertical columns to kilograms per square metre.

    Args:
        no2_molecules_per_cm2: TEMPO vertical-column values.

    Returns:
        NO2 mass columns in kilograms per square metre.
    """
    factor = CM2_PER_M2 * NO2_KG_PER_MOL / AVOGADRO_MOLECULES_PER_MOL
    return np.asarray(no2_molecules_per_cm2, dtype=np.float64) * factor


def _gaussian_with_linear_background(
    across_m: np.ndarray,
    line_density: float,
    width_m: float,
    shift_m: float,
    slope: float,
    intercept: float,
) -> np.ndarray:
    # Match the normalized Gaussian used by ddeq.csf
    gaussian = line_density / (width_m * np.sqrt(2.0 * np.pi))
    gaussian *= np.exp(-0.5 * ((across_m - shift_m) / width_m) ** 2)
    return gaussian + slope * across_m + intercept


def _source_wind(
    wind_u_mps: np.ndarray,
    wind_v_mps: np.ndarray,
    source_row: float,
    source_col: float,
) -> tuple[float, float]:
    # Bilinearly sample the aligned wind rasters at the fractional source pixel
    coordinates = np.array([[source_row], [source_col]], dtype=np.float64)
    wind_u = float(map_coordinates(wind_u_mps, coordinates, order=1, mode="nearest")[0])
    wind_v = float(map_coordinates(wind_v_mps, coordinates, order=1, mode="nearest")[0])
    return wind_u, wind_v


def _plume_coordinates(
    shape: tuple[int, int],
    source_row: float,
    source_col: float,
    wind_u_mps: float,
    wind_v_mps: float,
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Rotate cell centres into straight along-plume and across-plume coordinates
    rows, columns = np.indices(shape, dtype=np.float64)
    east_m = (columns - source_col) * cell_size_m
    north_m = (source_row - rows) * cell_size_m
    wind_speed = np.hypot(wind_u_mps, wind_v_mps)
    east_unit = wind_u_mps / wind_speed
    north_unit = wind_v_mps / wind_speed
    along_m = east_m * east_unit + north_m * north_unit
    across_m = -east_m * north_unit + north_m * east_unit
    return along_m, across_m


def _fit_line_density(
    across_m: np.ndarray,
    mass_column: np.ndarray,
    cell_size_m: float,
    maximum_crosswind_m: float,
) -> tuple[float, float]:
    # Fit a Gaussian plume enhancement and simultaneous linear background
    edge = np.abs(across_m) >= 0.75 * maximum_crosswind_m
    background = float(np.median(mass_column[edge])) if edge.any() else float(np.median(mass_column))
    width_start = 5.0 * cell_size_m
    enhancement = max(0.0, float(np.max(mass_column) - background))
    line_density_start = max(
        enhancement * width_start * np.sqrt(2.0 * np.pi),
        np.finfo(np.float64).eps,
    )
    shift_bound = min(10.0 * cell_size_m, maximum_crosswind_m)
    width_bound = min(50.0 * cell_size_m, maximum_crosswind_m)
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        parameters, covariance = curve_fit(
            _gaussian_with_linear_background,
            across_m,
            mass_column,
            p0=(line_density_start, width_start, 0.0, 0.0, background),
            bounds=(
                (0.0, 0.5 * cell_size_m, -shift_bound, -np.inf, -np.inf),
                (np.inf, width_bound, shift_bound, np.inf, np.inf),
            ),
            maxfev=20_000,
        )
    return float(parameters[0]), float(np.sqrt(covariance[0, 0]))


def _fit_source_flux(
    downwind_m: np.ndarray,
    flux_kg_per_second: np.ndarray,
    flux_precision: np.ndarray,
    wind_speed_mps: float,
    lifetime_seconds: float | None,
) -> tuple[float, float]:
    # Fit source flux with the fixed exponential lifetime used in CSF studies
    if lifetime_seconds is None:
        attenuation = np.ones_like(downwind_m)
    else:
        attenuation = np.exp(-downwind_m / (wind_speed_mps * lifetime_seconds))
    finite_precision = np.isfinite(flux_precision) & (flux_precision > 0.0)
    if finite_precision.any():
        weights = np.zeros_like(flux_precision)
        weights[finite_precision] = 1.0 / flux_precision[finite_precision] ** 2
    else:
        weights = np.ones_like(flux_kg_per_second)
    denominator = np.sum(weights * attenuation**2)
    emission = max(0.0, float(np.sum(weights * attenuation * flux_kg_per_second) / denominator))
    if finite_precision.any():
        precision = np.sqrt(1.0 / denominator)
    else:
        residual = flux_kg_per_second - emission * attenuation
        degrees_of_freedom = max(1, len(residual) - 1)
        residual_variance = float(np.sum(residual**2) / degrees_of_freedom)
        precision = np.sqrt(residual_variance / np.sum(attenuation**2))
    return emission, float(precision)


def estimate_cross_section_flux(
    no2_molecules_per_cm2: np.ndarray,
    quality: np.ndarray,
    wind_u_mps: np.ndarray,
    wind_v_mps: np.ndarray,
    source_row: float,
    source_col: float,
    config: CrossSectionFluxConfig = CrossSectionFluxConfig(),
) -> CrossSectionFluxEstimate | None:
    """Estimate one source's NOx emission rate from a TEMPO raster.

    Args:
        no2_molecules_per_cm2: Physical TEMPO NO2 vertical-column raster.
        quality: Boolean raster marking observed NO2 pixels.
        wind_u_mps: Eastward 80 m wind raster.
        wind_v_mps: Northward 80 m wind raster.
        source_row: Fractional source row in the north-up raster.
        source_col: Fractional source column in the raster.
        config: CSF geometry and physical assumptions.

    Returns:
        Source-level estimate, or ``None`` when quality requirements fail.
    """
    no2 = np.asarray(no2_molecules_per_cm2, dtype=np.float64)
    visible = np.asarray(quality, dtype=bool) & np.isfinite(no2)
    wind_u, wind_v = _source_wind(
        np.asarray(wind_u_mps, dtype=np.float64),
        np.asarray(wind_v_mps, dtype=np.float64),
        source_row,
        source_col,
    )
    wind_speed = float(np.hypot(wind_u, wind_v))
    if not config.minimum_wind_speed_mps <= wind_speed <= config.maximum_wind_speed_mps:
        return None

    along_m, across_m = _plume_coordinates(
        no2.shape,
        source_row,
        source_col,
        wind_u,
        wind_v,
        config.cell_size_m,
    )
    mass_column = tempo_no2_to_mass_column(no2)
    transect_centres = np.arange(
        config.cell_size_m,
        config.maximum_downwind_m + 0.5 * config.cell_size_m,
        config.cell_size_m,
    )
    distances: list[float] = []
    line_densities: list[float] = []
    line_density_precision: list[float] = []
    for centre in transect_centres:
        selected = (
            visible
            & (np.abs(along_m - centre) <= 0.5 * config.cell_size_m)
            & (np.abs(across_m) <= config.maximum_crosswind_m)
        )
        if selected.sum() < 5:
            continue
        try:
            line_density, precision = _fit_line_density(
                across_m[selected],
                mass_column[selected],
                config.cell_size_m,
                config.maximum_crosswind_m,
            )
        except (OptimizeWarning, RuntimeError, ValueError):
            continue
        if np.isfinite(line_density):
            distances.append(float(centre))
            line_densities.append(line_density)
            line_density_precision.append(precision)

    if len(distances) < config.minimum_transects:
        return None
    distance_array = np.asarray(distances)
    line_density_array = np.asarray(line_densities)
    line_density_precision_array = np.asarray(line_density_precision)
    flux = wind_speed * config.no2_to_nox_factor * line_density_array
    flux_precision = wind_speed * config.no2_to_nox_factor * line_density_precision_array
    emission, emission_precision = _fit_source_flux(
        distance_array,
        flux,
        flux_precision,
        wind_speed,
        config.nox_lifetime_seconds,
    )
    return CrossSectionFluxEstimate(
        emission_lb_per_hour=emission * KG_PER_SECOND_TO_LB_PER_HOUR,
        emission_precision_lb_per_hour=emission_precision * KG_PER_SECOND_TO_LB_PER_HOUR,
        wind_speed_mps=wind_speed,
        downwind_distance_m=distance_array,
        line_density_kg_per_m=line_density_array,
        flux_kg_per_second=flux,
        valid_transects=len(distance_array),
    )


def time_weighted_window_emissions(
    timestep_emission_lb_per_hour: np.ndarray,
    elapsed_hours: np.ndarray,
) -> tuple[float, float]:
    """Calculate the three-timestep early and late trapezoidal means.

    Args:
        timestep_emission_lb_per_hour: Five consecutive emission estimates.
        elapsed_hours: Four durations between consecutive observations.

    Returns:
        Elapsed-time-weighted early and late emission means.
    """
    interval_means = (timestep_emission_lb_per_hour[:-1] + timestep_emission_lb_per_hour[1:]) / 2.0
    early = np.average(interval_means[:2], weights=elapsed_hours[:2])
    late = np.average(interval_means[2:], weights=elapsed_hours[2:])
    return float(early), float(late)


def classify_emission_delta(
    delta_emission_lb_per_hour: float,
    threshold_lb_per_hour: float = STRATIFICATION_WINDOW_THRESHOLD,
) -> str:
    """Classify an early-to-late emission change with the dataset threshold.

    Args:
        delta_emission_lb_per_hour: Late mean minus early mean.
        threshold_lb_per_hour: Absolute boundary for increase and decrease.

    Returns:
        Decrease, steady, or increase.
    """
    if delta_emission_lb_per_hour <= -threshold_lb_per_hour:
        return "decrease"
    if delta_emission_lb_per_hour >= threshold_lb_per_hour:
        return "increase"
    return "steady"


def estimate_aoi_sequence_flux(
    no2_molecules_per_cm2: np.ndarray,
    quality: np.ndarray,
    wind_u_mps: np.ndarray,
    wind_v_mps: np.ndarray,
    elapsed_hours: np.ndarray,
    config: CrossSectionFluxConfig = CrossSectionFluxConfig(),
) -> CrossSectionFluxSequenceEstimate:
    """Estimate aggregate AOI flux at five timesteps from a fixed centreline.

    The centreline starts at the primary facility at the centre of the AOI.
    Transect fits may include enhancements from any facility in the raster.

    Args:
        no2_molecules_per_cm2: Five physical TEMPO NO2 rasters.
        quality: Five observed-pixel masks.
        wind_u_mps: Five eastward 80 m wind rasters.
        wind_v_mps: Five northward 80 m wind rasters.
        elapsed_hours: Four durations between consecutive observations.
        config: CSF geometry and physical assumptions.

    Returns:
        Per-timestep estimates and label-compatible temporal summaries.
    """
    estimates = [
        estimate_cross_section_flux(
            no2_molecules_per_cm2[timestep],
            quality[timestep],
            wind_u_mps[timestep],
            wind_v_mps[timestep],
            AOI_CENTRE_PIXEL,
            AOI_CENTRE_PIXEL,
            config,
        )
        for timestep in range(len(no2_molecules_per_cm2))
    ]
    emissions = np.asarray(
        [estimate.emission_lb_per_hour if estimate is not None else np.nan for estimate in estimates]
    )
    precision = np.asarray(
        [estimate.emission_precision_lb_per_hour if estimate is not None else np.nan for estimate in estimates]
    )
    wind_speed = np.asarray([estimate.wind_speed_mps if estimate is not None else np.nan for estimate in estimates])
    valid_transects = np.asarray(
        [estimate.valid_transects if estimate is not None else 0 for estimate in estimates],
        dtype=np.int64,
    )
    if np.isfinite(emissions).all():
        early, late = time_weighted_window_emissions(emissions, elapsed_hours)
        delta = late - early
        predicted_class = classify_emission_delta(delta)
    else:
        early = late = delta = float("nan")
        predicted_class = None
    return CrossSectionFluxSequenceEstimate(
        timestep_emission_lb_per_hour=emissions,
        timestep_precision_lb_per_hour=precision,
        timestep_wind_speed_mps=wind_speed,
        timestep_valid_transects=valid_transects,
        early_emission_lb_per_hour=early,
        late_emission_lb_per_hour=late,
        delta_emission_lb_per_hour=delta,
        predicted_class=predicted_class,
    )


def _elapsed_hours(record: pd.Series) -> np.ndarray:
    # Calculate the four observation intervals in hours
    timestamps = pd.to_datetime(record.loc[list(TIMESTEP_TIME_COLUMNS)], utc=True).to_numpy(dtype="datetime64[ns]")
    return (np.diff(timestamps) / np.timedelta64(1, "h")).astype(np.float64)


def _raster_path(serialized_path: object, dataset_dir: Path) -> Path:
    # Resolve a stored raster path against the dataset root
    path = Path(str(serialized_path))
    return path if path.is_absolute() else dataset_dir / path


def _estimate_record(
    record: pd.Series,
    dataset_dir: Path,
    config: CrossSectionFluxConfig,
) -> tuple[CrossSectionFluxSequenceEstimate, np.ndarray, np.ndarray]:
    # Load one physical raster sequence and run the fixed-centre AOI estimate
    path = _raster_path(record[RASTER_PATH_COL], dataset_dir)
    with np.load(path, allow_pickle=False) as bundle:
        no2 = np.asarray(bundle["no2"], dtype=np.float64)
        quality = np.asarray(bundle["no2_mask"], dtype=bool)
        estimate = estimate_aoi_sequence_flux(
            no2,
            quality,
            np.asarray(bundle["wind_u_80m_mps"], dtype=np.float64),
            np.asarray(bundle["wind_v_80m_mps"], dtype=np.float64),
            _elapsed_hours(record),
            config,
        )
    return estimate, no2, quality


def _balanced_sample_candidates(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    # Interleave fixed-seed class shuffles to keep the montage representative
    groups = {
        class_name: group.sample(frac=1.0, random_state=seed).reset_index()
        for class_name, group in frame.groupby(MODEL_TARGET_COL, sort=False)
    }
    records = []
    for offset in range(max(len(group) for group in groups.values())):
        for class_name in MODEL_CLASS_NAMES:
            group = groups.get(class_name)
            if group is not None and offset < len(group):
                records.append(group.iloc[offset])
    return pd.DataFrame(records)


def _prediction_row(record: pd.Series, estimate: CrossSectionFluxSequenceEstimate) -> dict[str, object]:
    # Flatten one estimate into an auditable CSV row
    row: dict[str, object] = {
        "record_index": int(record["index"]),
        "aoi_id": record["aoi_id"],
        "raster_bundle_path": record[RASTER_PATH_COL],
        "class_actual": record[MODEL_TARGET_COL],
        "class_predicted": estimate.predicted_class,
        "early_predicted_nox_lb_per_hour": estimate.early_emission_lb_per_hour,
        "late_predicted_nox_lb_per_hour": estimate.late_emission_lb_per_hour,
        "predicted_delta_nox_lb_per_hour": estimate.delta_emission_lb_per_hour,
    }
    for timestep, column in enumerate(TIMESTEP_NOX_COLUMNS):
        row[f"t{timestep}_actual_nox_lb_per_hour"] = float(record[column])
        row[f"t{timestep}_predicted_nox_lb_per_hour"] = float(
            estimate.timestep_emission_lb_per_hour[timestep]
        )
        row[f"t{timestep}_wind_speed_mps"] = float(estimate.timestep_wind_speed_mps[timestep])
        row[f"t{timestep}_valid_transects"] = int(estimate.timestep_valid_transects[timestep])
    return row


def _plot_sample_report(
    samples: list[tuple[pd.Series, CrossSectionFluxSequenceEstimate, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    # Plot five physical NO2 frames and CSF estimates for every sample
    figure, axes = plt.subplots(len(samples), 5, figsize=(18, 3.15 * len(samples)), constrained_layout=True)
    colour_map = plt.get_cmap("magma").copy()
    colour_map.set_bad("lightgray")
    for row_index, (record, estimate, no2, quality) in enumerate(samples):
        finite = no2[np.isfinite(no2) & quality]
        lower, upper = np.percentile(finite, (2.0, 98.0))
        for timestep, axis in enumerate(axes[row_index]):
            shown = np.where(quality[timestep], no2[timestep], np.nan)
            axis.imshow(shown, cmap=colour_map, vmin=lower, vmax=upper)
            axis.scatter(AOI_CENTRE_PIXEL, AOI_CENTRE_PIXEL, marker="+", s=60, linewidths=1.5, color="cyan")
            actual = float(record[TIMESTEP_NOX_COLUMNS[timestep]])
            predicted = estimate.timestep_emission_lb_per_hour[timestep]
            axis.set_title(f"t{timestep}  actual {actual:,.0f}\nCSF {predicted:,.0f} lb/hr", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(
            f"AOI {record['aoi_id']}\nactual: {record[MODEL_TARGET_COL]}\npred: {estimate.predicted_class}",
            fontsize=9,
        )
    figure.suptitle(
        "Straight-centerline aggregate CSF on validation samples\n"
        "cyan + = fixed AOI-center facility; gray = unavailable NO2",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_sample_report(
    *,
    split: str,
    sample_count: int,
    seed: int,
    output_path: str | Path,
    manifest_path: str | Path,
    dataset_dir: str | Path = DATASET_DIR,
    dataframe_dir: str | Path = DATASET_DF,
    config: CrossSectionFluxConfig = CrossSectionFluxConfig(),
) -> None:
    """Run CSF until a fixed-seed class-balanced sample report is complete.

    Args:
        split: Geographic dataset split to sample.
        sample_count: Number of valid sequences to include.
        seed: Sampling seed.
        output_path: Destination PNG path.
        manifest_path: Destination CSV path.
        dataset_dir: Root containing physical raster bundles.
        dataframe_dir: Directory containing split metadata.
        config: CSF geometry and physical assumptions.
    """
    frame = pd.read_csv(Path(dataframe_dir) / f"{split}_df.csv")
    candidates = _balanced_sample_candidates(frame, seed)
    selected = []
    attempted = 0
    for _, record in candidates.iterrows():
        attempted += 1
        estimate, no2, quality = _estimate_record(record, Path(dataset_dir), config)
        if estimate.predicted_class is None:
            continue
        selected.append((record, estimate, no2, quality))
        print(f"CSF sample {len(selected):,}/{sample_count:,} after {attempted:,} candidates")
        if len(selected) == sample_count:
            break
    if len(selected) != sample_count:
        raise RuntimeError(f"Only {len(selected)} valid CSF sequences found in {attempted} candidates")

    output = Path(output_path)
    manifest = Path(manifest_path)
    _plot_sample_report(selected, output)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_prediction_row(record, estimate) for record, estimate, _, _ in selected]).to_csv(
        manifest,
        index=False,
    )
    metadata = {
        "split": split,
        "seed": seed,
        "sample_count": sample_count,
        "candidate_records_attempted": attempted,
        "source_pixel": [AOI_CENTRE_PIXEL, AOI_CENTRE_PIXEL],
        "config": config.__dict__,
    }
    with manifest.with_suffix(".json").open("w") as destination:
        json.dump(metadata, destination, indent=2)
    print(f"Wrote CSF montage: {output}")
    print(f"Wrote CSF manifest: {manifest}")


def _parse_args() -> argparse.Namespace:
    # Parse the focused sample-report interface
    parser = argparse.ArgumentParser(description="Run straight-centerline aggregate CSF on sample sequences")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path(DATASET_DIR))
    parser.add_argument("--dataframe-dir", type=Path, default=Path(DATASET_DF))
    return parser.parse_args()


def main() -> None:
    """Run a fixed-seed CSF sample report from the command line."""
    args = _parse_args()
    create_sample_report(
        split=args.split,
        sample_count=args.sample_count,
        seed=args.seed,
        output_path=args.output,
        manifest_path=args.manifest_output,
        dataset_dir=args.dataset_dir,
        dataframe_dir=args.dataframe_dir,
    )


if __name__ == "__main__":
    main()
