# Setup and pipeline

The provided scripts are designed for UC Berkeley's
Savio HPC environment but can be easily adapted to a different compute environment.

## Environment

Requires Python 3.11, `uv`, and Linux x86-64. The lockfile uses PyTorch 2.3.1
with CUDA 11.8.

```bash
git clone https://github.com/pranav-walimbe/powerplant-modeling.git
cd powerplant-modeling
make setup
```

Add source credentials to the ignored repository-root `.env`:

```dotenv
CAMPD_API_KEY=your_campd_api_key
EARTHDATA_USERNAME=your_nasa_earthdata_username
EARTHDATA_PASSWORD=your_nasa_earthdata_password
```

HRRR downloads use the public NOAA archive without credentials.

## Configure paths

Replace the cluster paths in [`src/config.py`](../../src/config.py) with writable
locations before running commands.

| Config parameter | Contents |
|---|---|
| `EMISSIONS_BASE_DIR` | Raw and enriched CAMPD records |
| `TEMPO_BASE_DIR` | TEMPO downloads and mappings |
| `HRRR_DIR` | HRRR downloads |
| `STRAT_BASE_DIR` | Train/validation/test record CSVs |
| `DATASET_DIR` | Generated rasters, metadata, and persistent caches |
| `VIS_DIR` | Preprocessing diagnostics |
| `RUNS_DIR` | Training, search, and ablation outputs |

## 1. Collect source data

Set collection dates in `src/config.py`. For the paper's period:

| Config parameter | Value |
|---|---|
| `EMISSIONS_START_DATE` | `date(2023, 8, 1)` |
| `EMISSIONS_END_DATE` | `date(2026, 6, 30)` |
| `TEMPO_START_DATE` | `"2023-08-02 00:00:00"` |
| `TEMPO_END_DATE` | `"2026-06-30 23:59:59"` |
| `TEMPO_LEVEL` / `TEMPO_VERSION` | `"L2"` / `"V04"` |
| `HRRR_START_DATE` / `HRRR_END_DATE` | Inherit emissions dates unless changed |

The default end dates extend through the current date, subject to availability.
Set fixed dates for a reproducible pull.

```bash
uv run python -u src/data-scraping/scrape_tempo.py
uv run python -u src/data-scraping/scrape_hrrr.py --workers 4
uv run python -u src/data-scraping/scrape_emissions.py
uv run python -u src/data-scraping/scrape_locations.py
```

| Entry point | Parameters to supply |
|---|---|
| `scrape_tempo.py` | Config: `TEMPO_START_DATE`, `TEMPO_END_DATE`, `TEMPO_LEVEL`, `TEMPO_VERSION`, `TEMPO_BASE_DIR` |
| `scrape_hrrr.py` | Config: `HRRR_START_DATE`, `HRRR_END_DATE`, `HRRR_DIR`; optional CLI overrides: `--start-date`, `--end-date` (`YYYY-MM-DD`), `--output-dir`. Runtime: `--workers`, `--overwrite` |
| `scrape_emissions.py` | Config: `EMISSIONS_START_DATE`, `EMISSIONS_END_DATE`, `EMISSIONS_BASE_DIR` |
| `scrape_locations.py` | Config: `EMISSIONS_RECORDS_PARQUET` input, `FULL_DATA_PARQUET` output |

Run locations enrichment after emissions collection.

## 2. Build TEMPO mappings

```bash
uv run python -m preprocessing.tempo_mapping index --workers 4
uv run python -m preprocessing.tempo_mapping observations \
    --task-id 0 --task-count 1 --workers 4
```

| Argument / config | Use |
|---|---|
| `--workers` | Parallel processes; defaults to `NUM_CORES` |
| `--task-id`, `--task-count` | Required for `observations`; `0`, `1` covers all months in one invocation. For N tasks, run IDs `0` through `N-1` with `--task-count N` |
| `--overwrite` | Rebuild existing partitions; accepted by both subcommands |
| `TEMPO_DIR`, `FULL_DATA_PARQUET` | Source inputs |
| `TEMPO_GRANULE_MAPPING`, `TEMPO_AOI_MAPPING` | Mapping output directories |
| `IMG_RANGE` | AOI width in km; default `72` |
| `TEMPO_CELL_OVERLAP_FLOOR_KM2` | Minimum accepted footprint/cell overlap; default `0.0` |

Finish `index` before `observations`, and all observation tasks before stage 3.
For a Slurm job array, each array task supplies its zero-based ID as `--task-id`
and the total task count as `--task-count` to process a separate subset of months.

## 3. Create split records

```bash
uv run python -m preprocessing.stratify_plants
```

Set these in `src/config.py` before running:

| Config parameter | Default / use |
|---|---|
| `TEMPO_MIN_DELTA_MINUTES`, `TEMPO_MAX_DELTA_MINUTES` | `40`, `70`; allowed spacing between observations |
| `SEQUENCE_TIMESTEPS` | `5`; observations per sequence |
| `STRATIFICATION_WINDOW_THRESHOLD` | `250.0` lb/h; change-label threshold |
| `STRATIFICATION_AOI_FRACTION` | `0.50`; highest-scoring AOI fraction to retain |
| `TARGET_LABEL_MODE` | `"overlap_weighted"` |
| `MIN_CITY_POPULATION` | `500000`; major-city cutoff |
| `STRAT_BASE_DIR` | Destination for `train_records.csv`, `val_records.csv`, `test_records.csv` |

Optional diagnostic output paths can be supplied with `--diagnostic-output`,
`--monthly-prevalence-output`, `--solar-hour-prevalence-output`,
`--aoi-characteristics-plot`, `--aoi-characteristics-output`, and
`--aoi-selection-output`; defaults are under `VIS_DIR`.

## 4. Generate raster sequences

```bash
uv run python -m preprocessing.generate_dataset --split all --batch-size 500
```

**Requires Slurm's `squeue` even in direct mode.** On a machine without Slurm,
the active-job check must be adapted before this stage can run.
Generation replaces generated outputs and published metadata; wait until other
generation/training processes using this dataset have finished.
The caches store regridded TEMPO and aligned weather rasters so sequences sharing
the same AOI and observation time can reuse them without repeating preprocessing.

| CLI argument | Use |
|---|---|
| `--split` | `all`, `train`, `val`, or `test`; use `all` to build the complete dataset |
| `--batch-size` | Required; records staged together by sharded workers. Direct mode still requires the argument |
| `--shard-size` | Records per Slurm task; omit for direct generation, supply to launch a Slurm array from a login node |
| `--max-parallel-shards` | Concurrent array tasks; default `8` |
| `--workers-per-shard` | Processes/CPUs per task; default `8` |
| `--afterok-job-id` | Hold the shard array until this Slurm job succeeds |
| `--refresh-tempo` | Clear and rebuild the persistent TEMPO raster cache |
| `--refresh-weather` | Clear and rebuild the persistent weather raster cache |
| `--refresh-cache` | Clear and rebuild both caches |

| Config parameter | Default / use |
|---|---|
| `IMG_SIZE` | `24`; raster width/height in pixels |
| `MIN_PIXEL_CLOUD` | `0.20`; maximum pixel cloud fraction |
| `MIN_TIMESTEP_NO2_FINITE_FRACTION` | `0.85`; minimum finite NO2 coverage at every timestep |
| `NUM_CORES` | Direct-mode processes; reads `SLURM_CPUS_PER_TASK`, otherwise CPU count |
| `DATASET_DIR`, `DATASET_DF` | Generated data root and published split metadata |
| `DATASET_TEMPO_CACHE_DIR`, `DATASET_WEATHER_CACHE_DIR` | Persistent cache locations |

Refresh the affected cache after changing source data or regridding settings.
For a separate experiment, configure a new `DATASET_DIR` and regenerate mappings,
split records, and rasters as required by the changed settings.

## 5. Train and evaluate

```bash
uv run python -m modeling.train \
    --device cuda \
    --completed-raster-dir /tmp/no2-training/filled-rasters \
    --head-dim 48
```

This runs both the time-only MLP and ConvGRU. `--head-dim 48` selects the paper
setting; the training CLI default is `64`. Both models are evaluated on train,
validation, and test splits after checkpoint selection by validation macro F1.

### Runtime and storage

| Argument / config | Default / use |
|---|---|
| `--completed-raster-dir` | Required; temporary interpolated rasters below `/tmp`. Allow enough local disk space |
| `--device` | `auto`; accepts `auto`, `cpu`, `cuda` |
| `--batch-size` | `128`; reduce for GPU memory limits |
| `--workers` | `min(4, NUM_CORES)`; data-loader workers |
| `--prefetch-factor` | `2`; batches prefetched per worker |
| `--seed` | `42` |
| `DATASET_DIR`, `DATASET_DF` | Configured source rasters and split metadata |
| `RUNS_DIR` | Parent directory for timestamped run outputs |

Normalization and NO2 interpolation run automatically; no separate
AOI-normalization command is needed.

### Training parameters

Pass these to `uv run python -m modeling.train` to override the CLI defaults:

| Argument | Default | Controls |
|---|---|---|
| `--model-width` | `64` | Vision encoder / recurrent width |
| `--head-dim` | `64` | Classifier head width; use `48` for the paper setting |
| `--dropout` | `0.20` | Vision activation dropout |
| `--epochs` | `100` | Maximum vision epochs |
| `--seasonal-epochs` | `75` | Maximum time-only MLP epochs |
| `--learning-rate` | `6.0378112892784365e-5` | Vision learning rate |
| `--seasonal-learning-rate` | `1e-3` | MLP learning rate |
| `--weight-decay` | `1e-4` | Optimizer weight decay |
| `--gradient-clip-norm` | `5.0` | Gradient norm limit |
| `--scheduler-patience` | `5` | Validation-loss plateau patience |
| `--scheduler-factor` | `0.50` | Learning-rate reduction multiplier |
| `--early-stop-patience` | `10` | Epochs without checkpoint improvement before stopping |

Inspect `run_config.json` for the parameters used, `results.json` for metrics,
`checkpoints/` for selected models, and prediction CSVs / PNGs for diagnostics.

## 6. Optional hyperparameter search

```bash
uv run python -m modeling.optimize \
    --device cuda \
    --completed-raster-dir /tmp/no2-hpo/filled-rasters \
    --trials 24 --batch-size 128 --epochs 100 --workers 4 --seed 42
```

| Argument | Default / use |
|---|---|
| `--trials` | `24`; trials attempted in this invocation, including when resuming |
| `--study-name` | `classification-hpo`; retain the name when resuming |
| `--study-dir` | New timestamped directory under `RUNS_DIR`; supply an existing directory to resume |
| `--completed-raster-dir` | Required; temporary completed rasters below `/tmp` |
| `--device`, `--batch-size`, `--epochs`, `--workers`, `--prefetch-factor`, `--seed` | Same defaults as training |

Resume with 12 additional trials:

```bash
uv run python -m modeling.optimize \
    --device cuda \
    --completed-raster-dir /tmp/no2-hpo/filled-rasters \
    --study-dir /path/to/existing/study \
    --study-name classification-hpo --trials 12
```

To change the search space, edit constants in
[`optimize.py`](../../src/delta-model/modeling/optimize.py):

| Constant | Values |
|---|---|
| `LEARNING_RATE_RANGE` | `5e-6`–`2e-4`, log-uniform |
| `WEIGHT_DECAY_CHOICES` | `1e-4`, `1e-3` |
| `DROPOUT_CHOICES` | `0.2`, `0.4` |
| `MODEL_WIDTH_CHOICES` | `32`, `64`, `96` |
| `FIXED_HEAD_DIM` | `48` |

Search uses train/validation data only. Copy selected values from
`study_summary.json` into the final training command; search does not update
training defaults. The study directory also contains `trials.csv`, `study.db`,
`search_config.json`, and trial outputs.

## 7. Optional input ablations

```bash
uv run python -m modeling.ablation \
    --device cuda \
    --completed-raster-dir /tmp/no2-ablation/filled-rasters
```

- `--completed-raster-dir` is required; use a directory below `/tmp`.
- `--device` defaults to `cuda`; use `cpu` for CPU execution.
- Edit `ABLATIONS` in [`ablation.py`](../../src/delta-model/modeling/ablation.py)
  to change experiments. The runner executes the baseline and all listed input
  ablations; there is no CLI selector for individual experiments.
- Width `64`, head `48`, activation dropout `0.20`, tabular branch dropout `0.25`, and
  training controls are fixed in the module / imported defaults. Batch size,
  epochs, workers, and learning rate are not CLI options for this runner.
- Results, including `ablation_summary.txt`, are written under `RUNS_DIR`.

## Cluster launchers

Optional wrappers are in [`scripts/slurm/`](../../scripts/slurm/). Before using
them, replace `<USER>`, `<ACCOUNT>`, and `<EMAIL>`, update checkout/storage/log
paths, and check the requested cluster resources. Some wrappers also use SSH
and `mailx` for result emails; adapt those settings for your environment.
