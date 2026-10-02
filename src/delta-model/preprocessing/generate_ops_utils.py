"""Shared plant-operation matching utilities."""

from __future__ import annotations

import polars as pl
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    build_aoi_membership,
    build_aoi_spatial_frame,
    build_aois,
)

FACILITY_ID_COL = "facilityId"
UNIT_ID_COL = "unitId"
TIME_COL = "emissions_hour_utc"
LABEL_COL = "delta_category"
MEAN_OP_TIME_COL = "mean_unit_op_time"
PREVIOUS_MEAN_OP_TIME_COL = "previous_mean_unit_op_time"


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
        samples.select("sample_id", AOI_ID_COL, LABEL_COL, "t0_timestamp", "t4_timestamp")
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
