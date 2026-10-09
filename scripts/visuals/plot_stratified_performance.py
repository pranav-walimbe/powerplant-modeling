"""Generate the held-out stratified performance figure for the paper."""

import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import f1_score

from config import FULL_DATA_PARQUET

plt.switch_backend("Agg")


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPOSITORY_ROOT / "assets" / "paper" / "visuals"
OUTPUT_STEMS = (
    OUTPUT_DIR / "stratified-performance-monthly",
    OUTPUT_DIR / "stratified-performance-solar-hour",
    OUTPUT_DIR / "stratified-performance-aoi-nox",
    OUTPUT_DIR / "stratified-performance-census-region",
)
RUN_DIR = Path("/global/home/users/pranavwalimbe/model_runs/classification_only_zscore_20261007_202105")

MODEL_NAMES = ("Baseline MLP", "ConvGRU")
MODEL_FILES = ("seasonal_test_predictions.csv", "test_predictions.csv")
MODEL_COLORS = ("#0072B2", "#E69F00")
MODEL_MARKERS = ("o", "s")
MODEL_LINESTYLES = ("--", "-")

TRUE_CLASS_COLUMN = "class_true"
PREDICTED_CLASS_COLUMN = "class_predicted"
FIRST_TIMESTAMP_COLUMN = "t0_timestamp"
FINAL_TIMESTAMP_COLUMN = "t4_timestamp"
ALIGNMENT_COLUMNS = ("aoi_id", "raster_bundle_path", TRUE_CLASS_COLUMN)
PREDICTION_COLUMNS = (
    "aoi_id",
    "aoi_score",
    "lon",
    FIRST_TIMESTAMP_COLUMN,
    FINAL_TIMESTAMP_COLUMN,
    "raster_bundle_path",
    TRUE_CLASS_COLUMN,
    PREDICTED_CLASS_COLUMN,
)

CLASS_LABELS = np.arange(3)
AOI_HALF_NAMES = ("Lower 50%", "Upper 50%")
Y_LIMITS = (0.28, 0.72)
Y_TICKS = np.arange(0.3, 0.71, 0.1)
DEGREES_PER_SOLAR_HOUR = 15.0
PANEL_FIGURE_SIZE = (3.35, 2.55)

CENSUS_REGION_STATES = (
    ("Northeast", frozenset(("CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"))),
    ("Midwest", frozenset(("IN", "IL", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"))),
    (
        "South",
        frozenset(
            ("DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX")
        ),
    ),
    ("West", frozenset(("AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"))),
)
STATE_TO_CENSUS_REGION = {state: region for region, states in CENSUS_REGION_STATES for state in states}


def _load_predictions() -> dict[str, pd.DataFrame]:
    # Load both models on the same held-out records
    frames = {
        model_name: pd.read_csv(RUN_DIR / filename, usecols=PREDICTION_COLUMNS)
        for model_name, filename in zip(MODEL_NAMES, MODEL_FILES, strict=True)
    }
    reference = frames[MODEL_NAMES[0]].loc[:, ALIGNMENT_COLUMNS]
    for model_name in MODEL_NAMES[1:]:
        if not reference.equals(frames[model_name].loc[:, ALIGNMENT_COLUMNS]):
            raise ValueError(f"{model_name} predictions are not aligned with {MODEL_NAMES[0]}")
    return frames


def _macro_f1(frame: pd.DataFrame) -> float:
    # Calculate fixed-label macro F1 for one stratum
    return float(
        f1_score(
            frame[TRUE_CLASS_COLUMN],
            frame[PREDICTED_CLASS_COLUMN],
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        )
    )


def _accuracy(frame: pd.DataFrame) -> float:
    # Calculate record-level accuracy for one group
    return float((frame[TRUE_CLASS_COLUMN] == frame[PREDICTED_CLASS_COLUMN]).mean())


def _aoi_characteristics(reference: pd.DataFrame) -> pd.DataFrame:
    # Derive one set of static characteristics for each test AOI
    return reference.groupby("aoi_id", sort=True)[["aoi_score"]].first()


def _format_boundary(value: float) -> str:
    # Keep stratum ranges compact at paper scale
    if value >= 1_000:
        return f"{value:,.0f}"
    if np.isclose(value, round(value)):
        return f"{value:.0f}"
    return f"{value:.1f}"


def _aoi_half_scores(
    frames: dict[str, pd.DataFrame],
    values: pd.Series,
) -> tuple[list[str], dict[str, np.ndarray]]:
    # Split AOIs at the median and score their associated records
    percentiles = values.rank(method="average", pct=True)
    strata = pd.cut(
        percentiles,
        bins=(0.0, 0.5, 1.0),
        labels=AOI_HALF_NAMES,
        include_lowest=True,
    )
    reference = frames[MODEL_NAMES[0]]
    tick_labels: list[str] = []
    scores = {model_name: [] for model_name in MODEL_NAMES}
    for stratum_name in AOI_HALF_NAMES:
        aoi_ids = values.index[strata == stratum_name]
        mask = reference["aoi_id"].isin(aoi_ids)
        lower = _format_boundary(float(values.loc[aoi_ids].min()))
        upper = _format_boundary(float(values.loc[aoi_ids].max()))
        tick_labels.append(f"{stratum_name}\n{lower}-{upper}")
        for model_name in MODEL_NAMES:
            scores[model_name].append(_macro_f1(frames[model_name].loc[mask]))
    return tick_labels, {
        model_name: np.asarray(model_scores, dtype=np.float64) for model_name, model_scores in scores.items()
    }


def _monthly_scores(
    frames: dict[str, pd.DataFrame],
) -> tuple[list[str], dict[str, np.ndarray]]:
    # Score each calendar month on the held-out records
    reference = frames[MODEL_NAMES[0]]
    months = pd.to_datetime(reference[FINAL_TIMESTAMP_COLUMN], utc=True).dt.month
    month_numbers = sorted(months.unique())
    labels = [calendar.month_abbr[month] for month in month_numbers]
    scores = {
        model_name: np.asarray(
            [_accuracy(frame.loc[months == month]) for month in month_numbers],
            dtype=np.float64,
        )
        for model_name, frame in frames.items()
    }
    return labels, scores


def _solar_hour_scores(
    frames: dict[str, pd.DataFrame],
) -> tuple[list[str], dict[str, np.ndarray]]:
    # Score each midpoint local-solar-hour bin on the held-out records
    reference = frames[MODEL_NAMES[0]]
    start = pd.to_datetime(reference[FIRST_TIMESTAMP_COLUMN], utc=True)
    end = pd.to_datetime(reference[FINAL_TIMESTAMP_COLUMN], utc=True)
    midpoint_utc = start + (end - start) / 2
    midpoint_solar = midpoint_utc + pd.to_timedelta(
        reference["lon"] / DEGREES_PER_SOLAR_HOUR,
        unit="h",
    )
    solar_hours = midpoint_solar.dt.hour + midpoint_solar.dt.minute / 60.0 + midpoint_solar.dt.second / 3_600.0
    hour_bins = np.floor(solar_hours).astype(int)
    hours = sorted(hour_bins.unique())
    labels = [f"{hour:02d}:00" for hour in hours]
    scores = {
        model_name: np.asarray(
            [_accuracy(frame.loc[hour_bins == hour]) for hour in hours],
            dtype=np.float64,
        )
        for model_name, frame in frames.items()
    }
    return labels, scores


def _census_regions(aoi_ids: pd.Index) -> pd.Series:
    # Map each facility-centered AOI to its center facility state
    state_rows = (
        pl.scan_parquet(FULL_DATA_PARQUET)
        .filter(pl.col("facilityId").is_in(aoi_ids.to_list()))
        .select("facilityId", "stateCode")
        .group_by("facilityId")
        .agg(pl.col("stateCode").drop_nulls().first())
        .collect(engine="streaming")
    )
    states = pd.Series(
        state_rows["stateCode"].to_list(),
        index=state_rows["facilityId"].to_list(),
        dtype="string",
    ).reindex(aoi_ids)
    regions = states.map(STATE_TO_CENSUS_REGION)
    if regions.isna().any():
        missing = states.loc[regions.isna()].drop_duplicates().tolist()
        raise ValueError(f"Census region is unavailable for states: {missing}")
    return regions


def _census_region_scores(
    frames: dict[str, pd.DataFrame],
    aoi_ids: pd.Index,
) -> tuple[list[str], dict[str, np.ndarray]]:
    # Score every Census region represented by held-out AOIs
    regions = _census_regions(aoi_ids)
    represented = [region for region, _ in CENSUS_REGION_STATES if (regions == region).any()]
    reference = frames[MODEL_NAMES[0]]
    scores = {model_name: [] for model_name in MODEL_NAMES}
    tick_labels: list[str] = []
    for region in represented:
        region_aoi_ids = regions.index[regions == region]
        mask = reference["aoi_id"].isin(region_aoi_ids)
        tick_labels.append(f"{region}\n(n={len(region_aoi_ids)} AOIs)")
        for model_name in MODEL_NAMES:
            scores[model_name].append(_macro_f1(frames[model_name].loc[mask]))
    return tick_labels, {
        model_name: np.asarray(model_scores, dtype=np.float64) for model_name, model_scores in scores.items()
    }


def _plot_panel(
    axis: plt.Axes,
    tick_labels: list[str],
    scores: dict[str, np.ndarray],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    # Plot one paired model comparison
    positions = np.arange(len(tick_labels), dtype=np.float64)
    for model_name, color, marker, linestyle in zip(
        MODEL_NAMES,
        MODEL_COLORS,
        MODEL_MARKERS,
        MODEL_LINESTYLES,
        strict=True,
    ):
        axis.plot(
            positions,
            scores[model_name],
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
            marker=marker,
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=model_name,
        )
    axis.set(
        title=title,
        xlabel=x_label,
        ylabel=y_label,
        xticks=positions,
        xticklabels=tick_labels,
        ylim=Y_LIMITS,
        yticks=Y_TICKS,
    )
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_bar_panel(
    axis: plt.Axes,
    tick_labels: list[str],
    scores: dict[str, np.ndarray],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    # Plot paired bars for unordered Census-region categories
    positions = np.arange(len(tick_labels), dtype=np.float64)
    bar_width = 0.34
    offsets = (-bar_width / 2, bar_width / 2)
    for model_name, color, offset in zip(
        MODEL_NAMES,
        MODEL_COLORS,
        offsets,
        strict=True,
    ):
        bars = axis.bar(
            positions + offset,
            scores[model_name],
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            label=model_name,
        )
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=5.5)
    axis.set(
        title=title,
        xlabel=x_label,
        ylabel=y_label,
        xticks=positions,
        xticklabels=tick_labels,
        ylim=Y_LIMITS,
        yticks=Y_TICKS,
    )
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _save_panel(figure: plt.Figure, axis: plt.Axes, output_stem: Path) -> None:
    # Add the standalone legend and save both publication formats
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Generate four standalone stratified-performance figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    frames = _load_predictions()
    aoi = _aoi_characteristics(frames[MODEL_NAMES[0]])
    panels = (
        (
            *_monthly_scores(frames),
            "(a) Accuracy by month",
            "Month",
            "Accuracy",
        ),
        (
            *_solar_hour_scores(frames),
            "(b) Accuracy by midpoint local solar hour",
            "Midpoint local solar hour",
            "Accuracy",
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for output_stem, (tick_labels, scores, title, x_label, y_label) in zip(OUTPUT_STEMS[:2], panels, strict=True):
        figure, axis = plt.subplots(figsize=PANEL_FIGURE_SIZE)
        _plot_panel(
            axis,
            tick_labels,
            scores,
            title=title,
            x_label=x_label,
            y_label=y_label,
        )
        _save_panel(figure, axis, output_stem)

    aoi_labels, aoi_scores = _aoi_half_scores(frames, aoi["aoi_score"])
    figure, axis = plt.subplots(figsize=PANEL_FIGURE_SIZE)
    _plot_bar_panel(
        axis,
        aoi_labels,
        aoi_scores,
        title=r"(c) Macro F1 by AOI median NO$_x$",
        x_label=r"AOI median NO$_x$ group (lb hr$^{-1}$ range)",
        y_label="Macro F1",
    )
    _save_panel(figure, axis, OUTPUT_STEMS[2])

    region_labels, region_scores = _census_region_scores(frames, aoi.index)
    figure, axis = plt.subplots(figsize=PANEL_FIGURE_SIZE)
    _plot_bar_panel(
        axis,
        region_labels,
        region_scores,
        title="(d) Macro F1 by AOI-center Census region",
        x_label="Census region",
        y_label="Macro F1",
    )
    _save_panel(figure, axis, OUTPUT_STEMS[3])


if __name__ == "__main__":
    main()
