"""Partition AOI-hour emission records into train, validation, and test splits."""

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from preprocessing.diagnostic_plots import (
    CLASS_COLORS,
    CLASS_NAMES,
    plot_window_emissions_axes,
)
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    LABEL_MODE_COL,
    MAJOR_CITY_DIST_COL,
    add_aoi_bounds,
    add_major_city_distance,
    add_projected_coordinates,
    add_sequence_weather_paths,
    add_tempo_sequences,
    add_timestep_nox,
    add_window_average_targets,
    aggregate_aoi_hours,
    build_aoi_membership,
    build_aoi_spatial_frame,
    build_aois,
    calculate_activity_conditioned_aoi_features,
    calculate_operating_aoi_characteristics,
    cluster_aois,
    deterministic_weighted_sample,
    filter_groups_by_class_count,
    filter_usable_nox_measurements,
    usable_nox_measurement_expr,
)
from preprocessing.tempo_mapping import load_tempo_mapping

from config import (
    FULL_DATA_PARQUET,
    SEQUENCE_TIMESTEPS,
    STRAT_BASE_DIR,
    STRATIFICATION_AOI_FRACTION,
    STRATIFICATION_MINIMUM_RECORDS_PER_CLASS,
    STRATIFICATION_WINDOW_ABSOLUTE_FLOOR,
    STRATIFICATION_WINDOW_RELATIVE_FLOOR,
    TEST_RECORDS_CSV,
    TRAIN_RECORDS_CSV,
    VAL_RECORDS_CSV,
    VIS_DIR,
)

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42
MIN_STEADY_SELECTION_WEIGHT = 0.01
DELTA_CATEGORY_COL = "delta_category"
AOI_SCORE_COL = "aoi_score"
AOI_SCORE_PERCENTILE_COL = "aoi_score_percentile"
AVG_NOX_START_COL = "avg_nox_start"
AVG_NOX_END_COL = "avg_nox_end"
HYBRID_THRESHOLD_COL = "hybrid_window_threshold"
FINAL_TIMESTEP_INDEX = SEQUENCE_TIMESTEPS - 1
TIMESTEP_COLUMNS = [
    column
    for index in range(SEQUENCE_TIMESTEPS)
    for column in (
        f"t{index}_timestamp",
        f"timestep_age_hours_t{index}",
        f"t{index}_nox",
        f"no2_paths_t{index}",
        f"weather_path_t{index}",
    )
]
OUTPUT_COLUMNS = [
    AOI_ID_COL,
    AOI_SCORE_COL,
    AOI_SCORE_PERCENTILE_COL,
    "lat",
    "lon",
    "lat_min",
    "lat_max",
    "lon_min",
    "lon_max",
    MAJOR_CITY_DIST_COL,
    "num_coal_units",
    "num_ng_units",
    "num_units",
    "date",
    "hour",
    "emissions_hour_utc",
    "cluster",
    *TIMESTEP_COLUMNS,
    "label_delta_mins",
    "avg_heat_input",
    "avg_pwr_gen",
    "avg_coal_nox",
    AVG_NOX_START_COL,
    AVG_NOX_END_COL,
    HYBRID_THRESHOLD_COL,
    DELTA_CATEGORY_COL,
    LABEL_MODE_COL,
]
REQUIRED_COLUMNS = [
    "facilityId",
    "unitId",
    "lat",
    "lon",
    "date",
    "hour",
    "emissions_hour_utc",
    "noxMass",
    "opTime",
    "grossLoad",
    "heatInput",
    "noxMassMeasureFlg",
    "primaryFuelInfo",
    "attributePrimaryFuelInfo",
    "facility_nameplate_capacity_mw",
]

DEFAULT_DIAGNOSTIC_OUTPUT = Path(VIS_DIR) / "stratification_window_behavior.png"
DEFAULT_MONTHLY_PREVALENCE_OUTPUT = Path(VIS_DIR) / "stratification_monthly_prevalence.png"
DEFAULT_SOLAR_HOUR_PREVALENCE_OUTPUT = Path(VIS_DIR) / "stratification_solar_hour_prevalence.png"
DEFAULT_AOI_CHARACTERISTICS_PLOT = Path(VIS_DIR) / "stratification_aoi_characteristics.png"
DEFAULT_AOI_CHARACTERISTICS_OUTPUT = Path(VIS_DIR) / "stratification_aoi_characteristics.csv"
DEFAULT_AOI_SELECTION_OUTPUT = Path(VIS_DIR) / "stratification_aoi_selection.csv"
HISTOGRAM_QUANTILES = (0.01, 0.99)
AOI_CHARACTERISTIC_PANELS = (
    ("active_median_total_nox", "Active median total NOx", "log10(1 + lb/hr)", True),
    ("history_emitting_fraction", "Emitting-hour fraction", "Fraction", False),
    ("facility_count", "Facilities per AOI", "Count", False),
    ("active_mean_operating_units", "Active operating units", "Mean count", False),
    ("max_source_distance_km", "Maximum source distance", "km", False),
    ("largest_facility_capacity_share", "Dominant-facility capacity share", "Fraction", False),
)


def parse_args() -> argparse.Namespace:
    """Parse stratification command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-output", type=Path, default=DEFAULT_DIAGNOSTIC_OUTPUT)
    parser.add_argument("--monthly-prevalence-output", type=Path, default=DEFAULT_MONTHLY_PREVALENCE_OUTPUT)
    parser.add_argument("--solar-hour-prevalence-output", type=Path, default=DEFAULT_SOLAR_HOUR_PREVALENCE_OUTPUT)
    parser.add_argument(
        "--aoi-characteristics-plot",
        type=Path,
        default=DEFAULT_AOI_CHARACTERISTICS_PLOT,
    )
    parser.add_argument(
        "--aoi-characteristics-output",
        type=Path,
        default=DEFAULT_AOI_CHARACTERISTICS_OUTPUT,
    )
    parser.add_argument("--aoi-selection-output", type=Path, default=DEFAULT_AOI_SELECTION_OUTPUT)
    return parser.parse_args()


def _split_by_cluster(
    frame: pl.DataFrame,
    category_column: str | None = None,
) -> dict[str, pl.DataFrame]:
    # Greedily assign large clusters against record or class-specific targets
    category_names = CLASS_NAMES if category_column is not None else ()
    count_expressions = [pl.len().alias("records")]
    count_expressions.extend((pl.col(category_column) == category).sum().alias(category) for category in category_names)
    cluster_counts = (
        frame.group_by("cluster")
        .agg(count_expressions)
        .with_columns(pl.col("cluster").hash(seed=SPLIT_SEED).alias("_tie_breaker"))
        .sort(["records", "_tie_breaker"], descending=[True, False])
    )
    if cluster_counts.height < len(SPLIT_FRACTIONS):
        raise ValueError("At least three geographic clusters are required")

    target_columns = category_names or ("records",)
    totals = {column: float(cluster_counts[column].sum()) for column in target_columns}
    targets = {
        split: {column: totals[column] * fraction for column in target_columns}
        for split, fraction in SPLIT_FRACTIONS.items()
    }
    assigned = {split: {column: 0.0 for column in target_columns} for split in SPLIT_FRACTIONS}
    cluster_assignments: list[dict[str, object]] = []
    assigned_cluster_counts = {split: 0 for split in SPLIT_FRACTIONS}
    for index, cluster in enumerate(cluster_counts.iter_rows(named=True)):
        empty_splits = [split for split, count in assigned_cluster_counts.items() if count == 0]
        remaining_clusters = cluster_counts.height - index
        destinations = empty_splits if remaining_clusters == len(empty_splits) else SPLIT_FRACTIONS
        destination = min(
            destinations,
            key=lambda destination: sum(
                (
                    (
                        assigned[split][column]
                        + (float(cluster[column]) if split == destination else 0.0)
                        - targets[split][column]
                    )
                    / max(targets[split][column], 1.0)
                )
                ** 2
                for split in SPLIT_FRACTIONS
                for column in target_columns
            ),
        )
        cluster_assignments.append({"cluster": cluster["cluster"], "split": destination})
        assigned_cluster_counts[destination] += 1
        for column in target_columns:
            assigned[destination][column] += float(cluster[column])

    assignments = pl.DataFrame(
        cluster_assignments,
        schema={"cluster": frame.schema["cluster"], "split": pl.String},
    )
    splits = {
        split: frame.join(
            assignments.filter(pl.col("split") == split).select("cluster"),
            on="cluster",
            how="inner",
        )
        for split in SPLIT_FRACTIONS
    }
    for split, split_frame in splits.items():
        category_summary = ""
        if category_column is not None:
            counts = split_frame.group_by(category_column).len()
            by_category = dict(counts.iter_rows())
            category_summary = "; " + ", ".join(
                f"{category}={by_category.get(category, 0):,}" for category in category_names
            )
        print(
            f"[{split}] assigned {split_frame.height:,}/{frame.height:,} eligible records "
            f"({split_frame.height / frame.height:.1%}; target {SPLIT_FRACTIONS[split]:.1%})"
            f"{category_summary}"
        )
    return splits


def filter_stratification_rule(frame: pl.DataFrame) -> pl.DataFrame:
    """Assign classes from the first-three and last-three time-weighted means.

    Args:
        frame: Eligible records carrying time-weighted start and end NOx means.

    Returns:
        Records carrying the hybrid threshold and class.
    """
    midpoint = (pl.col(AVG_NOX_START_COL) + pl.col(AVG_NOX_END_COL)) / 2
    threshold = pl.max_horizontal(
        pl.lit(STRATIFICATION_WINDOW_ABSOLUTE_FLOOR),
        STRATIFICATION_WINDOW_RELATIVE_FLOOR * midpoint,
    )
    change = pl.col(AVG_NOX_END_COL) - pl.col(AVG_NOX_START_COL)
    return frame.with_columns(threshold.alias(HYBRID_THRESHOLD_COL)).with_columns(
        pl.when(change <= -pl.col(HYBRID_THRESHOLD_COL))
        .then(pl.lit("decrease"))
        .when(change >= pl.col(HYBRID_THRESHOLD_COL))
        .then(pl.lit("increase"))
        .otherwise(pl.lit("steady"))
        .alias(DELTA_CATEGORY_COL)
    )


def filter_aoi_class_floor(
    frame: pl.DataFrame,
    minimum_per_class: int = STRATIFICATION_MINIMUM_RECORDS_PER_CLASS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Retain AOIs with enough candidate records in every target class.

    Args:
        frame: Labeled candidate records.
        minimum_per_class: Required records in each class for one AOI.

    Returns:
        Eligible records and an AOI-level class-count audit.
    """
    return filter_groups_by_class_count(
        frame,
        AOI_ID_COL,
        DELTA_CATEGORY_COL,
        CLASS_NAMES,
        minimum_per_class,
    )


def _filter_metadata_eligibility(frame: pl.DataFrame) -> pl.DataFrame:
    # Apply non-raster candidate quality requirements
    return frame.filter(
        pl.all_horizontal([pl.col(f"t{index}_nox").is_finite() for index in range(SEQUENCE_TIMESTEPS)])
        & pl.col("avg_heat_input").is_finite()
        & pl.col("avg_pwr_gen").is_finite()
        & pl.col(MAJOR_CITY_DIST_COL).is_finite()
    )


def select_balanced_window_records(
    frame: pl.DataFrame,
    split: str,
    *,
    seed: int = SPLIT_SEED,
) -> pl.DataFrame:
    """Select equal deterministic samples from the three window-change classes.

    Args:
        frame: Eligible records carrying start and end NOx means.
        split: Split name used in progress and error messages.
        seed: Deterministic within-class ordering seed.

    Returns:
        All three buckets downsampled to the smallest bucket count.
    """
    counts = frame.group_by(DELTA_CATEGORY_COL).len().sort(DELTA_CATEGORY_COL)
    missing_classes = set(CLASS_NAMES).difference(counts[DELTA_CATEGORY_COL].to_list())
    if missing_classes:
        raise ValueError(f"[{split}] target classes have no records: {', '.join(sorted(missing_classes))}")
    records_per_class = int(counts["len"].min())
    selected = []
    identity_columns = (AOI_ID_COL, "emissions_hour_utc", f"t{FINAL_TIMESTEP_INDEX}_timestamp")
    for class_name in CLASS_NAMES:
        class_records = frame.filter(pl.col(DELTA_CATEGORY_COL) == class_name)
        if class_name == "steady":
            change = pl.col(AVG_NOX_END_COL) - pl.col(AVG_NOX_START_COL)
            weight = (1 - change.abs() / pl.col(HYBRID_THRESHOLD_COL)).clip(
                MIN_STEADY_SELECTION_WEIGHT,
                1.0,
            )
            class_records = deterministic_weighted_sample(
                class_records,
                records_per_class,
                weight,
                identity_columns,
                seed,
            )
        else:
            class_records = (
                class_records.with_columns(pl.struct(*identity_columns).hash(seed=seed).alias("_selection_tie_breaker"))
                .sort("_selection_tie_breaker", *identity_columns)
                .head(records_per_class)
                .drop("_selection_tie_breaker")
            )
        selected.append(class_records)
    balanced = pl.concat(selected).sort(AOI_ID_COL, "emissions_hour_utc")
    count_summary = ", ".join(f"{row[DELTA_CATEGORY_COL]}={row['len']:,}" for row in counts.iter_rows(named=True))
    print(
        f"[{split}] threshold=max({STRATIFICATION_WINDOW_ABSOLUTE_FLOOR:g}, "
        f"{STRATIFICATION_WINDOW_RELATIVE_FLOOR:.0%} of early-late midpoint); eligible {count_summary}; "
        f"selected {records_per_class:,} per class ({balanced.height:,} total)"
    )
    return balanced


def _plot_stratification_diagnostics(
    eligible_splits: dict[str, pl.DataFrame],
    balanced_splits: dict[str, pl.DataFrame],
    output_path: Path,
) -> None:
    # Show class-specific emissions trajectories and split composition
    split_names = tuple(SPLIT_FRACTIONS)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    plot_window_emissions_axes(
        {split_name: balanced_splits[split_name] for split_name in split_names},
        list(axes[0]),
    )
    for column_index, split_name in enumerate(split_names):
        eligible = eligible_splits[split_name]
        balanced = balanced_splits[split_name]
        category_counts = dict(eligible.group_by(DELTA_CATEGORY_COL).len().iter_rows())
        eligible_counts = np.array([category_counts.get(category, 0) for category in CLASS_NAMES])
        balanced_count = balanced.height // len(CLASS_NAMES)
        balanced_counts = np.full(len(CLASS_NAMES), balanced_count)
        positions = np.arange(len(CLASS_NAMES))
        width = 0.38
        bucket_axis = axes[1, column_index]
        before_bars = bucket_axis.bar(
            positions - width / 2,
            eligible_counts,
            width,
            label="Eligible",
            color=CLASS_COLORS,
            alpha=0.55,
        )
        after_bars = bucket_axis.bar(
            positions + width / 2,
            balanced_counts,
            width,
            label="Balanced",
            color=CLASS_COLORS,
        )
        bucket_axis.set_xticks(positions, CLASS_NAMES)
        bucket_axis.set_ylabel("Records")
        bucket_axis.grid(axis="y", alpha=0.2)
        bucket_axis.set_title(
            f"Eligible: {eligible.height:,} records, {eligible[AOI_ID_COL].n_unique():,} AOIs\n"
            f"Balanced: {balanced.height:,} records, {balanced[AOI_ID_COL].n_unique():,} AOIs"
        )
        bucket_axis.bar_label(before_bars, labels=[f"{count:,}" for count in eligible_counts], fontsize=8)
        bucket_axis.bar_label(after_bars, labels=[f"{count:,}" for count in balanced_counts], fontsize=8)
        if column_index == 0:
            bucket_axis.legend(loc="upper left")
    figure.suptitle(
        "Five-timestep emissions behavior by final class\n"
        f"Threshold: max({STRATIFICATION_WINDOW_ABSOLUTE_FLOOR:g} lb/hr, "
        f"{STRATIFICATION_WINDOW_RELATIVE_FLOOR:.0%} of early-late midpoint); lines show median and IQR"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _class_prevalence(frame: pl.DataFrame, group: pl.Expr, groups: list[int]) -> np.ndarray:
    # Calculate within-group class shares in a fixed group order
    counts = frame.with_columns(group.alias("_group")).group_by("_group", DELTA_CATEGORY_COL).len()
    lookup = {(int(row["_group"]), row[DELTA_CATEGORY_COL]): int(row["len"]) for row in counts.iter_rows(named=True)}
    values = np.array([[lookup.get((group_value, name), 0) for name in CLASS_NAMES] for group_value in groups])
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values, dtype=np.float64), where=totals > 0)


def _plot_class_prevalence(
    splits: dict[str, pl.DataFrame],
    group: pl.Expr,
    groups: list[int],
    tick_labels: list[str],
    xlabel: str,
    title: str,
    output_path: Path,
) -> None:
    # Plot within-bin class prevalence for each final split
    figure, axes = plt.subplots(1, len(SPLIT_FRACTIONS), figsize=(17, 5), sharey=True, constrained_layout=True)
    positions = np.arange(len(groups))
    for axis, (split_name, frame) in zip(axes, splits.items(), strict=True):
        prevalence = _class_prevalence(frame, group, groups)
        bottom = np.zeros(len(groups), dtype=np.float64)
        for class_index, (class_name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, strict=True)):
            values = 100 * prevalence[:, class_index]
            axis.bar(positions, values, bottom=bottom, color=color, label=class_name.title())
            bottom += values
        axis.set(
            title=f"{split_name.title()}: {frame.height:,} records, {frame[AOI_ID_COL].n_unique():,} AOIs",
            xlabel=xlabel,
            xticks=positions,
            xticklabels=tick_labels,
            ylim=(0, 100),
        )
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Class prevalence (%)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_monthly_class_prevalence(splits: dict[str, pl.DataFrame], output_path: Path) -> None:
    # Compare final class prevalence across calendar months
    groups = list(range(1, 13))
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    timestamp = pl.col(f"t{FINAL_TIMESTEP_INDEX}_timestamp")
    _plot_class_prevalence(
        splits,
        timestamp.dt.month(),
        groups,
        labels,
        "Month",
        "Monthly class prevalence after balancing",
        output_path,
    )


def _plot_solar_hour_class_prevalence(splits: dict[str, pl.DataFrame], output_path: Path) -> None:
    # Compare final class prevalence by local solar hour of the final scan
    timestamp = pl.col(f"t{FINAL_TIMESTEP_INDEX}_timestamp")
    solar_hour = (
        timestamp.dt.hour() + timestamp.dt.minute() / 60 + timestamp.dt.second() / 3_600 + pl.col("lon") / 15
    ) % 24
    groups = sorted(
        {
            int(value)
            for frame in splits.values()
            for value in frame.select(solar_hour.floor().cast(pl.Int8).alias("_solar_hour"))["_solar_hour"].to_list()
        }
    )
    _plot_class_prevalence(
        splits,
        solar_hour.floor().cast(pl.Int8),
        groups,
        [str(hour) for hour in groups],
        "Local solar hour",
        "Local-solar-hour class prevalence after balancing",
        output_path,
    )


def _rank_aoi_scores(all_aois: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    # Rank scored members of the facility-centered AOI set
    ranked = all_aois.select(AOI_ID_COL).join(scores, on=AOI_ID_COL, how="inner").sort(AOI_SCORE_COL, AOI_ID_COL)
    return ranked.with_row_index("aoi_score_rank", offset=1).with_columns(
        (pl.col("aoi_score_rank") / pl.len()).alias(AOI_SCORE_PERCENTILE_COL)
    )


def select_top_scored_aois(ranked_scores: pl.DataFrame, fraction: float) -> pl.DataFrame:
    """Select the highest-scoring share of mapped AOIs.

    Args:
        ranked_scores: Mapped AOIs carrying score ranks and percentiles.
        fraction: Selected share in the interval ``(0, 1]``.

    Returns:
        Deterministically selected AOI score rows.
    """
    selected_count = math.ceil(ranked_scores.height * fraction)
    return ranked_scores.sort(
        AOI_SCORE_COL,
        AOI_ID_COL,
        descending=[True, False],
    ).head(selected_count)


def _static_aoi_characteristics(relevant: pl.LazyFrame, membership: pl.DataFrame) -> pl.DataFrame:
    # Summarize source counts and facility capacity concentration
    static = (
        relevant.select("facilityId", "unitId")
        .unique()
        .join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL)
        .agg(
            pl.col("facilityId").n_unique().alias("facility_count"),
            pl.struct("facilityId", "unitId").n_unique().alias("unit_count"),
        )
        .collect(engine="streaming")
    )
    capacity = (
        relevant.group_by("facilityId")
        .agg(pl.col("facility_nameplate_capacity_mw").max().alias("_facility_capacity_mw"))
        .join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL)
        .agg(
            pl.col("_facility_capacity_mw").sum().alias("_total_capacity_mw"),
            pl.col("_facility_capacity_mw").max().alias("_largest_capacity_mw"),
        )
        .with_columns(
            pl.when(pl.col("_total_capacity_mw") > 0)
            .then(pl.col("_largest_capacity_mw") / pl.col("_total_capacity_mw"))
            .alias("largest_facility_capacity_share")
        )
        .select(AOI_ID_COL, "largest_facility_capacity_share")
        .collect(engine="streaming")
    )
    return static.join(capacity, on=AOI_ID_COL, how="inner")


def _geographic_aoi_characteristics(
    relevant: pl.LazyFrame,
    membership: pl.DataFrame,
    aois: pl.DataFrame,
) -> pl.DataFrame:
    # Measure the farthest member facility from each AOI center
    facility_points = add_projected_coordinates(
        relevant.select("facilityId", "lat", "lon").drop_nulls().unique(subset="facilityId").collect()
    ).select("facilityId", pl.col("x_m").alias("_facility_x_m"), pl.col("y_m").alias("_facility_y_m"))
    return (
        membership.join(facility_points, on="facilityId", how="inner")
        .join(
            aois.select(AOI_ID_COL, pl.col("x_m").alias("_aoi_x_m"), pl.col("y_m").alias("_aoi_y_m")),
            on=AOI_ID_COL,
            how="inner",
        )
        .with_columns(
            (
                (
                    (pl.col("_facility_x_m") - pl.col("_aoi_x_m")).pow(2)
                    + (pl.col("_facility_y_m") - pl.col("_aoi_y_m")).pow(2)
                ).sqrt()
                / 1_000
            ).alias("_source_distance_km")
        )
        .group_by(AOI_ID_COL)
        .agg(pl.col("_source_distance_km").max().alias("max_source_distance_km"))
    )


def calculate_aoi_characteristics(
    raw_records: pl.LazyFrame,
    membership: pl.DataFrame,
    aois: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate source and operating characteristics for one AOI subset.

    Args:
        raw_records: Full unit-hour emissions history.
        membership: Facility-to-AOI membership for the requested subset.
        aois: Requested AOI centers.

    Returns:
        One characteristic row per AOI.
    """
    relevant_facilities = membership["facilityId"].unique()
    relevant = raw_records.filter(pl.col("facilityId").is_in(relevant_facilities.implode()))
    static = _static_aoi_characteristics(relevant, membership)
    operating = calculate_operating_aoi_characteristics(relevant, membership)
    geometry = _geographic_aoi_characteristics(relevant, membership, aois)
    return static.join(
        operating,
        on=AOI_ID_COL,
        how="inner",
    ).join(geometry, on=AOI_ID_COL, how="inner")


def _plot_aoi_characteristics(characteristics: pl.DataFrame, output_path: Path) -> None:
    # Compare each selected AOI characteristic with the scored unselected cohort
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    cohorts = ((False, "Scored, unselected", "#999999"), (True, "Selected", "#3977af"))
    for axis, (column, title, xlabel, log_transform) in zip(
        axes.flat,
        AOI_CHARACTERISTIC_PANELS,
        strict=True,
    ):
        plotted: list[tuple[np.ndarray, str, str, float, float]] = []
        for selected, label, color in cohorts:
            values = characteristics.filter(pl.col("selected_for_stratification") == selected)[column].to_numpy()
            values = values[np.isfinite(values)]
            raw_median = float(np.median(values)) if values.size else np.nan
            if log_transform:
                values = np.log10(1 + np.clip(values, 0, None))
            if values.size:
                plotted.append((values, label, color, float(np.median(values)), raw_median))
        if not plotted:
            axis.set(title=title, xlabel=xlabel)
            axis.text(0.5, 0.5, "No finite values", transform=axis.transAxes, ha="center", va="center")
            continue
        combined = np.concatenate([values for values, _, _, _, _ in plotted])
        lower, upper = np.quantile(combined, HISTOGRAM_QUANTILES)
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        bins = np.linspace(lower, upper, 31)
        median_labels = []
        for values, label, color, median, raw_median in plotted:
            visible = values[(values >= lower) & (values <= upper)]
            axis.hist(visible, bins=bins, density=True, alpha=0.45, color=color, label=label)
            axis.axvline(median, color=color, linewidth=1.5, linestyle="--")
            median_labels.append(f"{label}: median {raw_median:,.2f}")
        axis.set(title=title, xlabel=xlabel, ylabel="Density")
        axis.grid(axis="y", alpha=0.2)
        axis.text(0.98, 0.96, "\n".join(median_labels), transform=axis.transAxes, ha="right", va="top", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=2)
    selected_count = characteristics.filter(pl.col("selected_for_stratification")).height
    figure.suptitle(
        f"AOI characteristics after active-median-NOx selection "
        f"({selected_count:,}/{characteristics.height:,} scored AOIs retained)",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _serialize_no2_paths(frame: pl.DataFrame) -> pl.DataFrame:
    # Encode every configured TEMPO granule list for CSV output
    return frame.with_columns(
        [
            pl.concat_str(pl.lit('["'), pl.col(f"no2_paths_t{index}").list.join('", "'), pl.lit('"]')).alias(
                f"no2_paths_t{index}"
            )
            for index in range(SEQUENCE_TIMESTEPS)
        ]
    )


def build_stratification_candidates(
    aoi_characteristics_plot: Path = DEFAULT_AOI_CHARACTERISTICS_PLOT,
    aoi_characteristics_output: Path = DEFAULT_AOI_CHARACTERISTICS_OUTPUT,
    aoi_selection_output: Path = DEFAULT_AOI_SELECTION_OUTPUT,
) -> pl.DataFrame:
    """Build eligible AOI-hour records for score-selected AOIs.

    Args:
        aoi_characteristics_plot: Destination for the AOI characteristics dashboard.
        aoi_characteristics_output: Destination for the underlying AOI table.
        aoi_selection_output: Destination for the complete AOI score-selection audit.

    Returns:
        Eligible records before geographic splitting.
    """
    source = pl.scan_parquet(FULL_DATA_PARQUET)
    raw_records = source.select(REQUIRED_COLUMNS).with_columns(pl.col("date").cast(pl.Date, strict=False))
    records = raw_records.pipe(filter_usable_nox_measurements).filter(pl.col("noxMass").is_finite())

    facilities = raw_records.select("facilityId", "lat", "lon").drop_nulls().unique(subset="facilityId").collect()
    all_aois = build_aois(facilities)
    all_spatial_aois = build_aoi_spatial_frame(all_aois)
    all_membership = build_aoi_membership(all_aois, facilities, all_spatial_aois)
    characteristics = calculate_aoi_characteristics(raw_records, all_membership, all_aois)
    nox_scores = characteristics.filter(
        pl.col("active_median_total_nox").is_finite() & (pl.col("active_median_total_nox") >= 0)
    ).select(AOI_ID_COL, pl.col("active_median_total_nox").alias(AOI_SCORE_COL))
    ranked_scores = _rank_aoi_scores(all_aois, nox_scores)
    selected_scores = select_top_scored_aois(ranked_scores, STRATIFICATION_AOI_FRACTION)
    selected_ids = selected_scores.select(AOI_ID_COL)
    selection_audit = (
        all_aois.select(AOI_ID_COL)
        .join(
            ranked_scores.select(AOI_ID_COL, AOI_SCORE_COL, AOI_SCORE_PERCENTILE_COL),
            on=AOI_ID_COL,
            how="left",
        )
        .join(selected_ids.with_columns(pl.lit(True).alias("selected_for_stratification")), on=AOI_ID_COL, how="left")
        .with_columns(
            pl.col(AOI_SCORE_COL).is_not_null().alias("has_aoi_score"),
            pl.col("selected_for_stratification").fill_null(False),
        )
        .sort(AOI_ID_COL)
    )
    aoi_selection_output.parent.mkdir(parents=True, exist_ok=True)
    selection_audit.write_csv(aoi_selection_output)
    aois = all_aois.join(selected_ids, on=AOI_ID_COL, how="inner")
    membership = all_membership.join(selected_ids, on=AOI_ID_COL, how="inner")
    characteristics = (
        characteristics.join(
            selected_ids.with_columns(pl.lit(True).alias("selected_for_stratification")),
            on=AOI_ID_COL,
            how="left",
        )
        .with_columns(pl.col("selected_for_stratification").fill_null(False))
        .sort(AOI_ID_COL)
    )
    aoi_characteristics_output.parent.mkdir(parents=True, exist_ok=True)
    characteristics.write_csv(aoi_characteristics_output)
    _plot_aoi_characteristics(characteristics, aoi_characteristics_plot)
    print(f"Saved AOI characteristics to {aoi_characteristics_plot} and {aoi_characteristics_output}")
    selected_features = calculate_activity_conditioned_aoi_features(raw_records, membership).join(
        selected_scores.select(AOI_ID_COL, AOI_SCORE_COL, AOI_SCORE_PERCENTILE_COL),
        on=AOI_ID_COL,
        how="inner",
    )
    spatial_aois = build_aoi_spatial_frame(aois)
    bounded_aois = add_major_city_distance(add_aoi_bounds(aois))
    observations = load_tempo_mapping()
    print(
        f"Selected {aois.height:,}/{ranked_scores.height:,} scored AOIs "
        f"from {all_aois.height:,} total AOIs by active median NOx; "
        f"{all_aois.height - ranked_scores.height:,} AOIs are unscored"
    )
    print(f"Saved complete AOI selection audit to {aoi_selection_output}")
    invalid_aoi_hours = (
        raw_records.filter(~usable_nox_measurement_expr() | ~pl.col("noxMass").is_finite())
        .join(membership.lazy(), on="facilityId", how="inner")
        .select(AOI_ID_COL, "emissions_hour_utc")
        .unique()
        .with_columns(pl.lit(True).alias("_has_invalid_nox"))
        .collect(engine="streaming")
    )
    hourly = (
        aggregate_aoi_hours(records, aois, membership, selected_features)
        .join(invalid_aoi_hours, on=[AOI_ID_COL, "emissions_hour_utc"], how="left")
        .filter(pl.col("_has_invalid_nox").is_null())
        .drop("_has_invalid_nox")
    )
    frame = hourly.join(cluster_aois(aois, spatial_aois), on=AOI_ID_COL, how="left")
    frame = add_tempo_sequences(frame, observations, SEQUENCE_TIMESTEPS)
    frame = frame.filter(pl.col(f"timestep_time_t{SEQUENCE_TIMESTEPS - 1}").is_not_null())
    frame = add_timestep_nox(frame, hourly, SEQUENCE_TIMESTEPS)
    frame = add_window_average_targets(frame)
    frame = frame.with_columns(pl.lit("time_weighted_first_last_three").alias(LABEL_MODE_COL))
    bounds = bounded_aois.select(AOI_ID_COL, "lat_min", "lat_max", "lon_min", "lon_max", MAJOR_CITY_DIST_COL)
    frame = add_sequence_weather_paths(
        frame.join(bounds, on=AOI_ID_COL, how="left"),
        SEQUENCE_TIMESTEPS,
    )
    frame = _filter_metadata_eligibility(frame).filter(
        pl.col(AVG_NOX_START_COL).is_finite() & pl.col(AVG_NOX_END_COL).is_finite()
    )
    return frame.rename({f"timestep_time_t{index}": f"t{index}_timestamp" for index in range(SEQUENCE_TIMESTEPS)})


def main() -> None:
    """Build stratified AOI-hour metadata splits for dataset generation."""
    args = parse_args()
    candidates = build_stratification_candidates(
        args.aoi_characteristics_plot,
        args.aoi_characteristics_output,
        args.aoi_selection_output,
    )
    frame = filter_stratification_rule(candidates)
    print(
        f"Labeled {frame.height:,} records with time-weighted window threshold "
        f"max({STRATIFICATION_WINDOW_ABSOLUTE_FLOOR:g} lb/hr, "
        f"{STRATIFICATION_WINDOW_RELATIVE_FLOOR:.0%} of early-late midpoint)"
    )
    frame, class_audit = filter_aoi_class_floor(frame)
    eligible_aoi_count = class_audit.filter(pl.col("meets_class_floor")).height
    print(
        f"Retained {eligible_aoi_count:,}/{class_audit.height:,} selected AOIs with at least "
        f"{STRATIFICATION_MINIMUM_RECORDS_PER_CLASS} records in every class"
    )
    eligible_splits = _split_by_cluster(frame, category_column=DELTA_CATEGORY_COL)
    splits = {
        split: select_balanced_window_records(split_frame, split) for split, split_frame in eligible_splits.items()
    }
    _plot_stratification_diagnostics(eligible_splits, splits, args.diagnostic_output)
    _plot_monthly_class_prevalence(splits, args.monthly_prevalence_output)
    _plot_solar_hour_class_prevalence(splits, args.solar_hour_prevalence_output)
    print(f"Saved window-behavior diagnostics to {args.diagnostic_output}")
    print(f"Saved monthly class prevalence to {args.monthly_prevalence_output}")
    print(f"Saved solar-hour class prevalence to {args.solar_hour_prevalence_output}")
    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    del frame
    # Project and write one split at a time so the copies never coexist
    for name, destination in (("train", TRAIN_RECORDS_CSV), ("val", VAL_RECORDS_CSV), ("test", TEST_RECORDS_CSV)):
        split = splits.pop(name)
        serialized = _serialize_no2_paths(split)
        serialized.select(OUTPUT_COLUMNS).write_csv(destination)


if __name__ == "__main__":
    main()
