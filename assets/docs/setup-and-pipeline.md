# Paper pipeline: setup and execution

This pipeline combines TEMPO Level 2 NO2 observations, EPA CAMPD hourly emissions
and operating records, HRRR meteorology, and plant metadata. It constructs
five-observation sequences and classifies their aggregate emissions trajectories
as `decrease`, `steady`, or `increase` using a ConvGRU and a time-only MLP baseline.

The provided shell scripts target **Savio, the University of California,
Berkeley's HPC cluster**. Savio uses **Slurm** to schedule compute jobs. Submit
batch scripts with `sbatch` from a login node; their `srun` commands execute the
work on allocated compute nodes. The two submission wrappers identified below
run with `bash` on a login node and submit their own Slurm jobs. For another
cluster, adapt the paths, Python module, resource requests, and mail setup.

## 1. Configure paths, credentials, and the environment

Clone the repository onto Savio. The shell-script path examples assume a checkout
at `/global/home/users/<USER>/no2-modeling`; change that path everywhere if your
checkout has a different name or location.

Before submitting any jobs, replace the literal placeholders in the shell scripts:

| Placeholder | Supply |
|---|---|
| `<USER>` | Your Savio Linux username, used in home, cache, log, and visualization paths |
| `<ACCOUNT>` | Your Slurm allocation account in `#SBATCH --account`; replace project-storage path components with the appropriate storage project as well |
| `<EMAIL>` | Your email address in both `#SBATCH --mail-user` and any `recipient` assignments |

These are text placeholders, not environment variables. Slurm does not expand
shell variables in `#SBATCH` directives. Your allocation account and Linux
username can differ, and your scratch-storage project may differ from the Slurm
account. Replace the full example scratch paths, including the `ddp/nox` suffix,
with writable paths for your project. Keep script paths consistent with
`src/config.py`.

Check your permitted account, partition, and QoS combinations on Savio:

```bash
sacctmgr -p show associations user="$USER" format=account,user,partition,qos
```

The supplied GPU jobs request one NVIDIA A5000 on `savio4_gpu` with four CPUs and
`a5k_gpu4_normal`. Verify access before using those requests. See Berkeley's
[Slurm submission guide](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/submitting-jobs/).

Set source-data, mapping, stratification, dataset, visualization, and model-run
paths in `src/config.py`. For the paper's study period, use TEMPO version `V04`
and August 2023 through June 2026: set `EMISSIONS_START_DATE` to August 1, 2023,
`EMISSIONS_END_DATE` to June 30, 2026, `TEMPO_START_DATE` to the beginning of the
available science record on August 2, 2023, and `TEMPO_END_DATE` to
`2026-06-30 23:59:59`. HRRR dates inherit the emissions dates in the config, but
`scripts/slurm/scrape_hrrr.sh` supplies its own date ranges; update its
`START_DATES` and `END_DATES` arrays and array-task count to cover the same period.
The default config otherwise requests data through the current date.

Retain the paper's sample settings: 72 km AOIs, 24 by 24 pixel rasters, five
timesteps spaced 40–70 minutes apart, a 250 lb/h emissions-change threshold,
the highest-scoring half of AOIs, cloud fraction at most 0.20, and at least 85%
finite NO2 coverage at every timestep.

You need `uv`, Savio access, an EPA CAMPD API key, and a NASA Earthdata account
with access to TEMPO. Add credentials to the repository's ignored `.env` file:

```dotenv
CAMPD_API_KEY=your_campd_api_key
EARTHDATA_USERNAME=your_nasa_earthdata_username
EARTHDATA_PASSWORD=your_nasa_earthdata_password
```

From the repository root on a login node, create the locked Python environment
and check the sources:

```bash
module load python/3.11.6-gcc-11.4.0
make setup
make check
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/src/delta-model"
mkdir -p logs
```

Create the configured output directories, including the visualization directory.
The `logs` directory must exist at the path in the `#SBATCH` directives before
submission. Batch jobs activate the existing `.venv`; they do not install it.

Slurm sends job-state notifications to `<EMAIL>`. Several scripts also send
diagnostic plots or summaries using SSH and `mailx` through the submission host.
Configure an email address and verify that this mail route works for your Savio
account, or adapt the artifact-email blocks. Supplying only `--mail-user` does
not change the separate artifact recipients.

## 2. Collect and enrich the source data

Submit collection jobs from the login node:

```bash
sbatch scripts/slurm/scrape_tempo_l2_v04.sh
sbatch scripts/slurm/scrape_hrrr.sh
emissions_job=$(sbatch --parsable scripts/slurm/refresh_campd_emissions.sh)
locations_job=$(sbatch --parsable --dependency="afterok:${emissions_job}" scripts/slurm/scrape_locations.sh)
```

The emissions job downloads CAMPD hourly records. The locations job reads those
records, adds facility metadata, and converts local standard time to UTC in
`emissions_hour_utc`. CAMPD standard offsets apply throughout the year. The
TEMPO and HRRR collection jobs can run independently of these two jobs.

Wait for all collection jobs and the locations job to finish successfully before
building mappings. Use `squeue -u "$USER"` to monitor queued/running jobs and
inspect their Slurm logs; absence from the queue alone does not establish success.

## 3. Build TEMPO mappings and geographic splits

Run the mapping submission wrapper on the login node:

```bash
bash scripts/slurm/submit_tempo_mapping.sh --overwrite
```

It submits a granule-index job followed by an AOI-observation mapping array with
an `afterok` dependency. The wrapper prints both job IDs. Submit stratification
after the whole observation array succeeds, replacing `OBSERVATION_ARRAY_JOB_ID`
with the printed array ID:

```bash
sbatch --dependency=afterok:OBSERVATION_ARRAY_JOB_ID scripts/slurm/stratify_plants.sh
```

Stratification ranks facility-centered AOIs by activity-conditioned median NOx
and retains the highest-scoring half. Labels compare time-weighted early and late
emissions windows over five observations. Overlapping AOIs stay in the same
geographic split, with approximately 70/15/15 allocations. Training classes are
downsampled to the smallest class. Validation and test retain all increase and
decrease records and sample steady records to match the larger change class.
The outputs are `train_records.csv`, `val_records.csv`, and `test_records.csv`
under `STRAT_BASE_DIR`, plus diagnostic figures and tables under `VIS_DIR`.

## 4. Generate the raster dataset

Once stratification succeeds, run this wrapper on a login node, not through
`sbatch` or from inside an existing Slurm job:

```bash
bash scripts/launch_dataset_generation.sh --batch-size 500
```

`--batch-size` is required; 500 is an example staging-batch size, which can be
adjusted for available memory and node-local storage. The wrapper defaults to
16,000 source records per shard and submits up to eight concurrent shard tasks
with eight workers each, followed by a dependent finalizer. To queue generation
behind stratification, add `--afterok STRATIFICATION_JOB_ID`.

Workers stage the TEMPO and HRRR inputs required by each batch under node-local
`/tmp`, reuse persistent regridding/weather caches, and write raster bundles.
They require at least 85% finite NO2 coverage at each timestep. The finalizer
validates every shard, publishes split CSVs under `DATASET_DF`, and emails a plot
of the retained trajectories and class counts. It does not rebalance classes
after raster filtering. Wait for the finalizer to succeed before training.

A new launch replaces disposable shards and published dataset metadata. Do not
launch generation while another generation run or model job uses those outputs.
Optional arguments include:

```bash
bash scripts/launch_dataset_generation.sh --batch-size 500 --shard-size 12000 -- --split train
```

The paper workflow uses all splits, which is the default. Arguments after `--`
go to the Python CLI. Use `--refresh-tempo`, `--refresh-weather`, or
`--refresh-cache` there when cached rasters must be rebuilt after source or
preprocessing changes.

## 5. Train and evaluate the paper models

Submit the training job after dataset finalization:

```bash
sbatch scripts/slurm/train_model.sh
```

The job stages split metadata into job-local `/tmp`, fits normalization statistics
on training data only, and fills missing NO2 pixels by separable bilinear
interpolation into temporary memory-mapped arrays. It computes transport
residuals and trains:

- The baseline MLP using four cyclic time features: sine/cosine of local solar
  hour and day of year.
- The ConvGRU using five-observation NO2 and meteorological raster sequences,
  pixel-validity information, transport residuals, and the same time features.

The launcher supplies the selected paper configuration: width 64, batch size 128,
activation dropout 0.20, learning rate approximately `6.04e-5`, weight decay
`1e-4`, gradient-norm limit 5, and validation macro-F1 model selection. It allows
up to 100 ConvGRU epochs and 75 baseline epochs, with early stopping. Training
evaluates the selected models on train, validation, and test splits and writes
checkpoints, normalization statistics, metrics, predictions, and plots under a
timestamped directory in `RUNS_DIR`. The launcher emails result plots afterward.

The separate AOI-normalization launcher is not required for this training path;
the classifier computes its own training-split normalization.

## 6. Hyperparameter search and ablations

The main training launcher already contains the selected hyperparameters. To
rerun selection, submit the validation-only search before the final training run:

```bash
sbatch scripts/slurm/optimize_model.sh
```

The search varies model width, learning rate, weight decay, and activation
dropout, selecting the highest validation macro-F1. It writes `study_summary.json`,
`trials.csv`, trial checkpoints, and a resumable study database under `RUNS_DIR`.
Copy the selected settings into `train_model.sh` before rerunning final training;
the search does not update that script automatically. `HPO_TRIALS` controls the
trial count (default 24).

Run the input ablation study on the finalized dataset with:

```bash
sbatch scripts/slurm/run_model_ablations.sh
```

This writes results under `RUNS_DIR` and emails an `ablation_summary.txt` file.
The collection, mapping, generation, training, search, and ablation jobs require
access to the configured shared storage; a repository checkout alone does not
contain the processed modeling dataset.
