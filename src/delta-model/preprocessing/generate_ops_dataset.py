"""Build an operations-focused evaluation manifest from generated samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from preprocessing.generate_dataset_utils import write_csv_atomic
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    build_aoi_membership,
    build_aoi_spatial_frame,
    build_aois,
)

from config import DATASET_DF, FULL_DATA_PARQUET, OPS_DATASET_CSV

SOURCE_SPLIT_COL = "source_split"
SAMPLE_ID_COL = "sample_id"
RASTER_PATH_COL = "raster_bundle_path"
FACILITY_ID_COL = "facilityId"
UNIT_ID_COL = "unitId"
TIME_COL = "emissions_hour_utc"
LABEL_COL = "delta_category"
MEAN_OP_TIME_COL = "mean_unit_op_time"
PREVIOUS_MEAN_OP_TIME_COL = "previous_mean_unit_op_time"
SPLITS = ("val", "test")
CLASS_NAMES = ("decrease", "steady", "increase")


def parse_args() -> argparse.Namespace:
    """Parse operations-dataset generation options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-sample-count", type=int, required=True)
    return parser.parse_args()


def load_generated_samples(dataframe_dir: str | Path = DATASET_DF) -> pl.DataFrame:
    """Load finalized validation and test records.

    Args:
        dataframe_dir: Directory containing generated split CSV files.

    Returns:
        Combined records with split provenance and stable sample identifiers.
    """
    directory = Path(dataframe_dir)
    frames = [
        pl.read_csv(directory / f"{split}_df.csv", try_parse_dates=True).with_columns(
            pl.lit(split).alias(SOURCE_SPLIT_COL)
        )
        for split in SPLITS
    ]
    return pl.concat(frames, how="vertical").with_row_index(SAMPLE_ID_COL)


def map_facilities_to_selected_aois(
    source: pl.LazyFrame,
    samples: pl.DataFrame,
) -> pl.DataFrame:
    """Map source facilities into the AOIs represented by the samples.

    Args:
        source: CAMPD records carrying facility coordinates.
        samples: Modeling samples with AOI identifiers.

    Returns:
        Facility-to-AOI containment pairs for selected AOIs.
    """
    facilities = (
        source.select(FACILITY_ID_COL, "lat", "lon")
        .drop_nulls()
        .unique(subset=FACILITY_ID_COL, keep="first")
        .collect(engine="streaming")
    )
    selected_ids = samples.select(AOI_ID_COL).unique()
    selected_aois = build_aois(facilities).join(selected_ids, on=AOI_ID_COL, how="inner")
    return build_aoi_membership(
        selected_aois,
        facilities,
        build_aoi_spatial_frame(selected_aois),
    ).select(AOI_ID_COL, FACILITY_ID_COL)


def build_plant_transitions(
    source: pl.LazyFrame,
    membership: pl.DataFrame,
) -> pl.DataFrame:
    """Find exact zero and nonzero crossings in plant operating time.

    Args:
        source: CAMPD unit-hour records.
        membership: Facility-to-selected-AOI containment pairs.

    Returns:
        Consecutive plant-hour start and stop events attached to their AOIs.
    """
    selected_facilities = membership.select(FACILITY_ID_COL).unique()
    plant_hours = (
        source.select(FACILITY_ID_COL, UNIT_ID_COL, TIME_COL, "opTime")
        .filter(pl.col("opTime").is_finite() & pl.col("opTime").is_between(0.0, 1.0, closed="both"))
        .join(selected_facilities.lazy(), on=FACILITY_ID_COL, how="inner")
        .group_by(FACILITY_ID_COL, TIME_COL)
        .agg(
            pl.col("opTime").mean().alias(MEAN_OP_TIME_COL),
            pl.col(UNIT_ID_COL).n_unique().alias("plant_unit_count"),
        )
        .sort(FACILITY_ID_COL, TIME_COL)
        .with_columns(
            pl.col(MEAN_OP_TIME_COL).shift(1).over(FACILITY_ID_COL).alias(PREVIOUS_MEAN_OP_TIME_COL),
            pl.col(TIME_COL).shift(1).over(FACILITY_ID_COL).alias("previous_hour"),
        )
        .filter((pl.col(TIME_COL) - pl.col("previous_hour")) == pl.duration(hours=1))
        .with_columns(
            pl.when((pl.col(PREVIOUS_MEAN_OP_TIME_COL) == 0) & (pl.col(MEAN_OP_TIME_COL) > 0))
            .then(pl.lit("start"))
            .when((pl.col(PREVIOUS_MEAN_OP_TIME_COL) > 0) & (pl.col(MEAN_OP_TIME_COL) == 0))
            .then(pl.lit("stop"))
            .alias("event_direction")
        )
        .filter(pl.col("event_direction").is_not_null())
        .collect(engine="streaming")
    )
    return plant_hours.join(membership, on=FACILITY_ID_COL, how="inner")


def build_sample_transition_hours(samples: pl.DataFrame) -> pl.DataFrame:
    """Expand each five-scan sample into hourly transition boundaries.

    Args:
        samples: Modeling samples with oldest and newest scan times.

    Returns:
        Sample and transition-hour keys within each observation window.
    """
    return (
        samples.select(SAMPLE_ID_COL, AOI_ID_COL, LABEL_COL, "t0_timestamp", "t4_timestamp")
        .with_columns(pl.col("t0_timestamp").dt.truncate("1h").alias("_t0_hour"))
        .with_columns(
            pl.when(pl.col("t0_timestamp") == pl.col("_t0_hour"))
            .then(pl.col("_t0_hour"))
            .otherwise(pl.col("_t0_hour") + pl.duration(hours=1))
            .alias("_first_transition_hour")
        )
        .with_columns(
            pl.datetime_ranges(
                pl.col("_first_transition_hour"),
                pl.col("t4_timestamp").dt.truncate("1h"),
                interval="1h",
                closed="both",
            ).alias(TIME_COL)
        )
        .explode(TIME_COL, empty_as_null=True)
        .filter(pl.col(TIME_COL).is_not_null())
    )


def build_ops_candidates(samples: pl.DataFrame, transitions: pl.DataFrame) -> pl.DataFrame:
    """Select one directionally valid five-timestep sample per operation episode.

    Args:
        samples: Generated validation and test samples.
        transitions: Plant-hour transitions mapped to sample AOIs.

    Returns:
        Unique raster samples carrying operational-event audit columns.
    """
    episodes = transitions.group_by(AOI_ID_COL, TIME_COL, "event_direction").agg(
        pl.col(FACILITY_ID_COL).n_unique().alias("ops_event_facility_count")
    )
    sample_metadata = samples.select(
        SAMPLE_ID_COL,
        SOURCE_SPLIT_COL,
        "t2_timestamp",
        RASTER_PATH_COL,
    )
    links = (
        build_sample_transition_hours(samples)
        .join(episodes, on=[AOI_ID_COL, TIME_COL], how="inner")
        .join(sample_metadata, on=SAMPLE_ID_COL, how="left")
        .with_columns((pl.col(TIME_COL) - pl.col("t2_timestamp")).abs().alias("_event_t2_distance"))
    )
    episode_keys = (SOURCE_SPLIT_COL, AOI_ID_COL, TIME_COL, "event_direction")
    canonical = (
        links.sort(*episode_keys, "_event_t2_distance", RASTER_PATH_COL, SAMPLE_ID_COL)
        .unique(subset=episode_keys, keep="first", maintain_order=True)
        .filter(
            ((pl.col("event_direction") == "start") & pl.col(LABEL_COL).is_in(("increase", "steady")))
            | ((pl.col("event_direction") == "stop") & pl.col(LABEL_COL).is_in(("decrease", "steady")))
        )
    )
    sample_events = (
        canonical.sort(SAMPLE_ID_COL, "_event_t2_distance", TIME_COL, "event_direction")
        .group_by(SAMPLE_ID_COL, maintain_order=True)
        .agg(
            pl.col(TIME_COL).first().alias("ops_nearest_event_hour"),
            pl.col("event_direction").first().alias("ops_nearest_event_direction"),
            pl.len().alias("ops_episode_count"),
            (pl.col("event_direction") == "start").sum().alias("ops_start_episode_count"),
            (pl.col("event_direction") == "stop").sum().alias("ops_stop_episode_count"),
            pl.col("ops_event_facility_count").sum(),
        )
    )
    return samples.join(sample_events, on=SAMPLE_ID_COL, how="inner")


def filter_eligible_aois(candidates: pl.DataFrame, min_sample_count: int) -> pl.DataFrame:
    """Retain every candidate from AOIs meeting both change-class minimums.

    Args:
        candidates: Deduplicated directionally valid operation samples.
        min_sample_count: Required increase and decrease samples per AOI.

    Returns:
        Candidate rows from qualifying AOIs.
    """
    eligible = (
        candidates.group_by(SOURCE_SPLIT_COL, AOI_ID_COL)
        .agg(
            (pl.col(LABEL_COL) == "increase").sum().alias("increase_samples"),
            (pl.col(LABEL_COL) == "decrease").sum().alias("decrease_samples"),
        )
        .filter((pl.col("increase_samples") >= min_sample_count) & (pl.col("decrease_samples") >= min_sample_count))
        .select(SOURCE_SPLIT_COL, AOI_ID_COL)
    )
    return candidates.join(eligible, on=[SOURCE_SPLIT_COL, AOI_ID_COL], how="inner")


def format_summary(samples: pl.DataFrame, min_sample_count: int, output_path: str | Path) -> str:
    """Format a tabular text summary for the completed dataset.

    Args:
        samples: Final operations evaluation records.
        min_sample_count: Applied increase and decrease minimum.
        output_path: CSV destination reported in the summary.

    Returns:
        Human-readable generation summary.
    """
    lines = [
        "Operations evaluation dataset",
        f"Minimum increase/decrease samples per AOI: {min_sample_count}",
        f"CSV: {output_path}",
        "",
        "scope\taois\tdecrease\tsteady\tincrease\ttotal",
    ]
    for scope in (*SPLITS, "pooled"):
        frame = samples if scope == "pooled" else samples.filter(pl.col(SOURCE_SPLIT_COL) == scope)
        counts = {name: frame.filter(pl.col(LABEL_COL) == name).height for name in CLASS_NAMES}
        lines.append(
            "\t".join(
                (
                    scope,
                    f"{frame[AOI_ID_COL].n_unique():,}",
                    f"{counts['decrease']:,}",
                    f"{counts['steady']:,}",
                    f"{counts['increase']:,}",
                    f"{frame.height:,}",
                )
            )
        )
    return "\n".join(lines)


def generate_ops_dataset(
    min_sample_count: int,
    *,
    dataframe_dir: str | Path = DATASET_DF,
    emissions_path: str | Path = FULL_DATA_PARQUET,
    output_path: str | Path = OPS_DATASET_CSV,
) -> tuple[pl.DataFrame, str]:
    """Generate and persist the operations-focused evaluation manifest.

    Args:
        min_sample_count: Required increase and decrease samples per AOI.
        dataframe_dir: Directory containing generated validation and test CSVs.
        emissions_path: Enriched CAMPD unit-hour parquet path.
        output_path: Destination for the modeling-ready CSV.

    Returns:
        Final records and their text summary.
    """
    samples = load_generated_samples(dataframe_dir)
    source = pl.scan_parquet(emissions_path)
    membership = map_facilities_to_selected_aois(source, samples)
    transitions = build_plant_transitions(source, membership)
    candidates = build_ops_candidates(samples, transitions)
    selected = (
        filter_eligible_aois(candidates, min_sample_count)
        .sort(SOURCE_SPLIT_COL, AOI_ID_COL, "ops_nearest_event_hour", RASTER_PATH_COL)
        .drop(SAMPLE_ID_COL)
    )
    write_csv_atomic(selected, Path(output_path))
    return selected, format_summary(selected, min_sample_count, output_path)


def main() -> None:
    """Generate the configured operations evaluation manifest."""
    args = parse_args()
    _, summary = generate_ops_dataset(args.min_sample_count)
    print(summary)


if __name__ == "__main__":
    main()
