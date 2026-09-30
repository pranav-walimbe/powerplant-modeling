# Dataset design

Each record joins five TEMPO scans, hourly HRRR fields, plant metadata, and one
three-class emissions-change label for a 72 km area of interest (AOI).

## Contract

| Component | Rule |
|---|---|
| Target column | `delta_category` |
| Classes | `decrease`, `steady`, `increase` |
| Split unit | Cluster of overlapping 72 km AOIs |
| Split target | About 70% train, 15% validation, 15% test by AOIs and candidate records |
| Metadata sampling | Balance training; retain held-out changes and fill with near-zero steady records |
| Raster quality | At least 85% finite NO2 per timestep |
| Model selection | Validation split only; test remains frozen |

## Record selection

The pipeline applies these steps in order:

1. Build the complete facility-centered AOI set. For each AOI-hour, calculate
   mean unit operating time and total usable NOx emissions. Retain hours at or
   above the AOI's median operating time, then score the AOI by its median NOx
   over those higher-activity hours. Rank AOIs by this emissions-only score
   with AOI ID as the deterministic tie-breaker and retain the highest-scoring
   half. No fuel-type or plant-characteristic value enters the score.
   Stratification saves a complete AOI selection audit and characteristic plot.
2. Aggregate usable CAMPD measurements for the selected AOIs by UTC hour. Add
   unit counts, major-city distance, and full-history heat-input and generation
   averages calculated over the same higher-activity AOI-hours.
3. Match five consecutive TEMPO scans whose adjacent timestamps are 40 to 70
   minutes apart. Store raster scans as `t0_timestamp` through `t4_timestamp`.
4. At each raster timestamp, linearly interpolate the CAMPD NOx rate between
   the surrounding UTC-hour values. Store these five interpolated rates as
   `t0_nox` through `t4_nox`.
5. Calculate elapsed-time-weighted trapezoidal means over `t0` through `t2`
   and `t2` through `t4`. Store them as `avg_nox_start` and `avg_nox_end`.
6. Assign classes from `avg_nox_end - avg_nox_start` using a 250 lb/hr
   absolute threshold.
7. Assign each overlap cluster to one split with a deterministic optimization
   targeting the 70/15/15 ratio for both AOIs and candidate records.
8. Downsample each training class to the smallest training-class count with a
   fixed seed. For validation and test, retain every increase and decrease
   record, then sample enough steady records to reach each split's target size.
   Define those target sizes from the selected training size and the 70/15/15
   proportions. Use deterministic weighted sampling for held-out steady records
   with weight `(1 - |window_change| / 250)^8`, floored at `1e-9`. This heavily
   favors records nearest zero change.
9. Generate rasters and reject records that fail coverage or source-file checks.

Raster quality control can change class counts after stratification. The
finalizer emails the retained class trajectories and does not rebalance the
surviving records.

## Label construction

For three scans at times `a`, `b`, and `c`, the elapsed-time-weighted mean is:

```text
weighted_mean(a:b:c) =
    ((b-a) * (NOx(a)+NOx(b))/2 + (c-b) * (NOx(b)+NOx(c))/2) / (c-a)
```

```text
avg_nox_start = weighted_mean(t0:t1:t2)
avg_nox_end = weighted_mean(t2:t3:t4)
window_change = avg_nox_end - avg_nox_start
hybrid_window_threshold = 250
```

Changes at or below the negative threshold are decreases. Changes at or above
the positive threshold are increases, and values between them are steady. The
pipeline writes both time-weighted means, the threshold, and the class to the
split metadata. The classifier consumes `delta_category` without recreating it
from a continuous value.

Metadata records this target construction as
`label_mode=time_weighted_first_last_three`. The label and the causal raster
sequence both end at `t4`; no post-label observation is included.

## Stored inputs

Each raster bundle stores five arrays with shape `5 x 24 x 24`:

| Array | Contents |
|---|---|
| `no2` | Area-weighted NO2 on accepted TEMPO support |
| `no2_mask` | Binary NO2 support mask |
| `temperature_2m_k` | HRRR temperature at AOI cell centers |
| `wind_u_80m_mps` | Geographic eastward HRRR wind |
| `wind_v_80m_mps` | Geographic northward HRRR wind |

Stratification metadata stores the activity-conditioned median-NOx score and percentile,
coal, natural-gas, and total unit counts. It also stores major-city distance
and activity-conditioned averages for heat input, generation, and coal NOx.
The AOI score is selection metadata and does not enter the model. The metadata
does not store nameplate capacity or normalized NOx-change targets.

## Raster checks and publication

The regridder accepts a TEMPO value when its quality flag is zero, cloud
fraction is at most 0.20, its value and geometry are valid, and its footprint
overlaps an output cell. Missing cells remain `NaN` in the stored raster and
zero in `no2_mask`.

Each timestep needs at least 85% finite NO2 cells. A 3 by 3 window around the
cell with the largest modeled-unit count needs full support. Unit-weighted
distance to the AOI center breaks source-cell ties.

The generator records cloud, quality, uncertainty, and coverage diagnostics in
the companion metadata. These diagnostics do not enter the classifiers.

Workers write disposable shards against persistent TEMPO and HRRR caches. The
finalizer validates all source outcomes and publishes split CSVs with raster
paths relative to the dataset root. One failed shard blocks publication.
