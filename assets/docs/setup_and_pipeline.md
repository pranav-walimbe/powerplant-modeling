# Setup and pipeline

This guide covers source-data collection, sequence construction, training and
evaluation of the ConvGRU and baseline MLP, hyperparameter search, and ablations.
Commands run from the repository root using the Python entry points.

## Setup

### Environment

Requires **Python 3.11**, [`uv`](https://docs.astral.sh/uv/), and Linux x86-64.
The locked environment includes PyTorch 2.3.1 from its CUDA 11.8 package index.
GPU examples below use `--device cuda`; use `--device cpu` for CPU execution.

```bash
git clone https://github.com/pranav-walimbe/powerplant-modeling.git
cd powerplant-modeling
make setup
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/src/delta-model"
```

Activate the environment and set `PYTHONPATH` in each new shell used to run the
commands. Python modules are under `src/delta-model`; collectors are invoked by
their file paths.

### Credentials and source data

Add credentials to the ignored `.env` file in the repository root:

```dotenv
CAMPD_API_KEY=your_campd_api_key
EARTHDATA_USERNAME=your_nasa_earthdata_username
EARTHDATA_PASSWORD=your_nasa_earthdata_password
```

- [NASA Earthdata](https://www.earthdata.nasa.gov/data/catalog/larc-cloud-tempo-no2-l2-v04): TEMPO version 4 Level 2 NO2 observations and quality information.
- [EPA CAMPD](https://campd.epa.gov/): hourly unit emissions, operating records, and facility metadata; API access requires a key.
- [NOAA HRRR archive](https://registry.opendata.aws/noaa-hrrr-pds/): temperature and wind fields; no account is required for the public archive.

The processed modeling dataset is available from the corresponding author upon
request and is not included in the repository. The commands below describe how
to construct it from the source data.

### Paths and study settings

Edit [`src/config.py`](../../src/config.py) before running the pipeline. Its
defaults contain the original experiment's cluster paths; replace them with
writable locations on your system.

| Settings | Purpose |
|---|---|
| `EMISSIONS_BASE_DIR` | Raw and enriched CAMPD records |
| `TEMPO_BASE_DIR` | TEMPO downloads and mapping outputs |
| `HRRR_DIR` | HRRR downloads |
| `STRAT_BASE_DIR` | Train, validation, and test record lists |
| `DATASET_DIR` | Generated raster bundles, metadata, and caches |
| `VIS_DIR` | Diagnostic plots and tables |
| `RUNS_DIR` | Model checkpoints, metrics, predictions, and search results |

Derived paths are built from these roots. `NO2_DATASET_DIR` and `NO2_DATASET_DF`
can override the dataset root and metadata directory through environment
variables. Keep all stages pointed at the same dataset. Create configured output
directories as needed, including `VIS_DIR` and `RUNS_DIR`.

For the paper's study period, use:

| Setting | Value |
|---|---|
| `TEMPO_VERSION` | `"V04"` |
| `EMISSIONS_START_DATE` | `date(2023, 8, 1)` |
| `EMISSIONS_END_DATE` | `date(2026, 6, 30)` |
| `TEMPO_START_DATE` | `"2023-08-02 00:00:00"` |
| `TEMPO_END_DATE` | `"2026-06-30 23:59:59"` |

HRRR dates inherit the emissions dates unless overridden on its CLI. The default
configuration otherwise requests data through the current date, subject to
source availability, so set fixed dates when reproducing the study period.

Retain the paper's sequence settings: 72 km AOIs, 24 × 24 pixel rasters, five
observations spaced 40–70 minutes apart, a 250 lb/h change threshold, the
highest-scoring half of AOIs, cloud fraction at most 0.20, and at least 85%
finite NO2 coverage at every timestep.

## Usage

Run the following stages in order, waiting for each command to finish
successfully before starting its dependent stage. Worker counts and batch sizes
are examples; adjust them for available CPU cores, memory, and storage.

### 1. Collect and enrich source data

```bash
python -u src/data-scraping/scrape_tempo.py

python -u src/data-scraping/scrape_hrrr.py \
    --start-date 2023-08-01 \
    --end-date 2026-06-30 \
    --workers 4

python -u src/data-scraping/scrape_emissions.py
python -u src/data-scraping/scrape_locations.py
```

Run locations enrichment after emissions collection: it reads the hourly
emissions records, adds facility metadata, and writes `FULL_DATA_PARQUET` with
UTC-aligned timestamps. CAMPD timestamps use local standard time, without
daylight-saving adjustments. TEMPO and HRRR downloads can run independently of
the emissions/enrichment pair.

### 2. Build TEMPO mappings and geographic splits

```bash
python -m preprocessing.tempo_mapping index --workers 4

python -m preprocessing.tempo_mapping observations \
    --task-id 0 \
    --task-count 1 \
    --workers 4

python -m preprocessing.stratify_plants
```

`--task-id 0 --task-count 1` processes all mapping months in one invocation.
Indexing must finish before observation mapping; mapping must finish before
stratification. Add `--overwrite` to either mapping command when replacing
existing partitions.

Stratification selects AOIs, constructs five-observation emissions-change
labels, and keeps overlapping AOIs in the same geographic split. It writes
`train_records.csv`, `val_records.csv`, and `test_records.csv` under
`STRAT_BASE_DIR`, plus diagnostics under `VIS_DIR`. Training classes are
downsampled to the smallest class; validation/test retain all change records and
sample steady records to match their larger change class.

### 3. Generate raster sequences

```bash
python -m preprocessing.generate_dataset \
    --split all \
    --batch-size 500
```

**Current portability limitation:** even this direct generation mode calls
Slurm's `squeue` to check for active generation/training jobs before replacing
dataset outputs. On a machine without Slurm, this stage currently fails at that
check. The scheduler check must be adapted before this command can run there;
the documentation does not imply that the existing implementation is fully
independent of Slurm.

Without `--shard-size`, the command uses the direct generation path rather than
submitting a Slurm array. `--batch-size` is required by the CLI; the staging-batch
behavior is used by the sharded workers. The direct path uses `NUM_CORES` from
the config for parallel processing.

Generation reuses persistent TEMPO and weather caches, writes raster bundles,
filters samples by finite NO2 coverage, and publishes split metadata under
`DATASET_DF`. It does not rebalance classes after raster filtering. New runs
replace generated rasters and published metadata, so do not regenerate while
another process uses that dataset.

Use `--refresh-tempo`, `--refresh-weather`, or `--refresh-cache` when cached
rasters must be rebuilt after changing source data or preprocessing.

### 4. Train and evaluate

```bash
python -m modeling.train \
    --device cuda \
    --completed-raster-dir /tmp/no2-training/filled-rasters \
    --batch-size 128 \
    --epochs 100 \
    --seasonal-epochs 75 \
    --workers 4 \
    --model-width 64 \
    --head-dim 48 \
    --dropout 0.20 \
    --learning-rate 6.0378112892784365e-5 \
    --weight-decay 1e-4 \
    --seed 42
```

This trains both the time-only MLP and ConvGRU using the selected paper
hyperparameters. It fits normalization on training data, fills missing NO2
pixels by separable bilinear interpolation, and computes transport residuals.
The classifier computes its own training-split normalization; the separate
AOI-normalization script is not required for this path.

`--completed-raster-dir` is required and must be under `/tmp`. It holds temporary
interpolated rasters; provide sufficient local disk space. Persistent source
rasters and split metadata are read from the configured dataset paths.

Models are selected by validation macro-F1 with early stopping, then evaluated
on training, validation, and test splits. A timestamped directory under
`RUNS_DIR` contains:

- ConvGRU and MLP checkpoints under `checkpoints/`.
- `normalization_stats.json`, `run_config.json`, and `results.json`.
- Prediction CSVs and performance/training plots.

### 5. Optional hyperparameter search

```bash
python -m modeling.optimize \
    --device cuda \
    --completed-raster-dir /tmp/no2-hpo/filled-rasters \
    --trials 24 \
    --batch-size 128 \
    --workers 4 \
    --seed 42
```

The validation-only search varies model width, learning rate, weight decay, and
activation dropout. It writes `study_summary.json`, `trials.csv`, trial
checkpoints, and `study.db` under a timestamped search directory in `RUNS_DIR`.
Use `--study-dir` to resume an existing study. Apply the selected settings to
the final training command; the search does not update training defaults.

### 6. Optional input ablations

```bash
python -m modeling.ablation \
    --device cuda \
    --completed-raster-dir /tmp/no2-ablation/filled-rasters
```

This retrains the model with individual inputs removed and writes the ablation
results, including `ablation_summary.txt`, under `RUNS_DIR`.

## Savio and Slurm launchers

We ran the paper's experiments on **UC Berkeley's Savio HPC cluster using
Slurm**, with one NVIDIA RTX A5000 GPU for model training. The shell scripts in
[`scripts/slurm/`](../../scripts/slurm/) reproduce that execution environment;
they are optional wrappers around the Python stages described above.

- Replace `<USER>` with your Linux username, `<ACCOUNT>` with the appropriate
  Slurm allocation/project, and `<EMAIL>` with your email address. Check both
  `#SBATCH --mail-user` and separate `recipient` assignments.
- Adapt checkout, shared-storage, output, and log paths to match `src/config.py`.
  A scratch-storage project can differ from the Slurm allocation account.
- Create log directories before submission and verify account, partition, QoS,
  GPU, and CPU access against your cluster's configuration.
- The mapping and dataset-generation wrappers run on a login node and submit
  dependent jobs; batch jobs run through `sbatch` and `srun`.
- Several wrappers email diagnostics through SSH and `mailx`. Configure that
  route or adapt those blocks. Direct Python commands do not send these emails.

See Berkeley's [Slurm submission guide](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/submitting-jobs/)
for cluster-specific execution details.
