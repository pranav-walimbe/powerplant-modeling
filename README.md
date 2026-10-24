<h1 align="center">Detecting Changes in Power Plant NO<sub>x</sub> Emissions Using Deep Learning and TEMPO Satellite Imagery</h1>

<p align="center">
  Pranav Walimbe · Hikari Murayama · Ron Cohen<br>
  University of California, Berkeley
</p>

<p align="center">
  <a href="assets/paper/paper.pdf"><b>Paper PDF</b></a> ·
  <a href="assets/docs/setup-and-pipeline.md"><b>Setup &amp; pipeline</b></a> ·
  <a href="#results"><b>Results</b></a> ·
  <a href="#citation"><b>Citation</b></a> ·
  <a href="LICENSE"><b>CC BY 4.0</b></a>
</p>

**Learn short-term emissions changes from five consecutive TEMPO observations.**

- **Task:** classify emissions trajectories as **decreasing**, **steady**, or **increasing**.
- **Inputs:** satellite NO2 rasters, meteorology, transport residuals, and cyclic time features.
- **Evaluation:** 22,466 held-out sequences covering 116 active U.S. power plants.
- **Test macro-F1:** **0.56** for ConvGRU, compared with **0.45** for the time-only MLP.

<p align="center">
  <a href="assets/paper/visuals/sample-construction.pdf">
    <img src="assets/readme/sample-construction.png" width="1000" alt="Example decrease-labeled sequence showing five NO2 observations, wind fields, surface temperature, transport residuals, and cyclic time features.">
  </a>
</p>

*One input sequence: five satellite observations and meteorological fields, four transport residuals, and four time features. Click a figure to open its PDF.*

## Results

**Macro-F1 across geographically separated training, validation, and test splits:**

| Model | Training | Validation | Test |
|---|---:|---:|---:|
| Time-only baseline MLP | 0.53 | 0.53 | 0.45 |
| **ConvGRU** | **0.62** | **0.60** | **0.56** |

<p align="center">
  <a href="assets/paper/visuals/classification-performance-test.pdf">
    <img src="assets/readme/classification-performance-test.png" width="1000" alt="Test-set F1 and recall by emissions-change class, and macro-F1, comparing the baseline MLP with ConvGRU.">
  </a>
</p>

- ConvGRU improves test F1 across all three classes; its largest advantage is for **steady** samples.
- **Transport residuals** contribute the most in the input ablation study.
- **Decreases remain difficult:** test recall is 0.47 for ConvGRU and 0.53 for the baseline.

## Method

1. **Align observations and emissions.** Regrid TEMPO NO2 and HRRR fields into 24 × 24 pixel rasters over 72 × 72 km plant-centered areas of interest (AOIs).
2. **Build five-observation sequences.** Derive emissions-change labels from the aggregate CEMS emissions of facilities within each AOI, using a 250 lb/h change threshold.
3. **Account for plume transport.** Compare each observed NO2 raster with the preceding plume's predicted transport and decay to construct residuals.
4. **Classify the trajectory.** Encode observations and residuals with separate branches, update a ConvGRU state, and fuse the final representation with time features.

<p align="center">
  <a href="assets/paper/visuals/convgru-architecture.pdf">
    <img src="assets/readme/convgru-architecture.png" width="1000" alt="ConvGRU architecture with observation and transport-residual encoders, four recurrent transitions, spatial pooling, and late fusion with a tabular time-feature branch.">
  </a>
</p>

- **724,939 trainable parameters** in the ConvGRU; **291** in the baseline MLP.
- Normalization uses **training data only**; model selection uses **validation macro-F1**.
- Predictions describe **aggregate AOI emissions changes**. They do not identify individual unit dispatch or establish the cause of a change.

## Data

**Study period: August 2023–June 2026.**

| Source | Data used | Access |
|---|---|---|
| TEMPO | Version 4 Level 2 tropospheric NO2 columns and retrieval-quality information | [NASA Earthdata](https://www.earthdata.nasa.gov/data/catalog/larc-cloud-tempo-no2-l2-v04) |
| EPA CEMS | Hourly unit emissions, operating records, and facility metadata | [EPA CAMPD](https://campd.epa.gov/) |
| HRRR | 2 m temperature and 80 m wind components | [NOAA public archive](https://registry.opendata.aws/noaa-hrrr-pds/) |

| Split | Sequences | AOIs | Active plants |
|---|---:|---:|---:|
| Training | 70,307 | 469 | 530 |
| Validation | 20,790 | 100 | 109 |
| Test | 22,466 | 98 | 116 |
| **Total** | **113,563** | **667** | **755** |

- **Processed modeling dataset:** available from the corresponding author upon request; contact details are in the [paper](assets/paper/paper.pdf).
- Raw source data are public. The processed dataset is **not included** in this repository.

## Getting started

The supplied job scripts target **Berkeley's Savio HPC cluster**, which uses **Slurm**. The locked environment requires **Python 3.11**, `uv`, and Linux x86-64; the paper's training launcher requests one NVIDIA RTX A5000 GPU.

**Before running:**

- Replace **`<USER>`**, **`<ACCOUNT>`**, and **`<EMAIL>`** in the shell scripts.
- Configure storage paths and the study dates in [`src/config.py`](src/config.py).
- Add NASA Earthdata and EPA CAMPD credentials to the ignored `.env` file.
- Confirm your Savio account, partition, and QoS access, and create the configured log/output directories.

**Install and check from the repository root on Savio:**

```bash
module load python/3.11.6-gcc-11.4.0
make setup
make check
```

**Run the workflow in order:**

| Stage | Entry points |
|---|---|
| Collect and enrich data | `scripts/slurm/scrape_tempo_l2_v04.sh`, `scrape_hrrr.sh`, `refresh_campd_emissions.sh`, `scrape_locations.sh` |
| Build mappings and splits | `scripts/slurm/submit_tempo_mapping.sh`, `stratify_plants.sh` |
| Generate raster sequences | `scripts/launch_dataset_generation.sh --batch-size 500` |
| Train and evaluate | `scripts/slurm/train_model.sh` |
| Rerun hyperparameter search | `scripts/slurm/optimize_model.sh` |
| Run input ablations | `scripts/slurm/run_model_ablations.sh` |

**Use the [full setup and pipeline guide](assets/docs/setup-and-pipeline.md) for commands, dependencies, placeholder substitutions, and output locations.** The mapping and dataset-generation wrappers run on a login node; other job scripts are submitted with `sbatch`. Wait for upstream jobs and dataset finalization before training.

## Repository guide

| Location | Contents |
|---|---|
| [`src/data-scraping/`](src/data-scraping/) | TEMPO, CAMPD, HRRR, and facility-data collection |
| [`src/delta-model/preprocessing/`](src/delta-model/preprocessing/) | Mapping, geographic stratification, regridding, and dataset generation |
| [`src/delta-model/modeling/`](src/delta-model/modeling/) | ConvGRU, MLP baseline, transport model, training, evaluation, HPO, and ablations |
| [`scripts/slurm/`](scripts/slurm/) | Savio batch jobs and submission helpers |
| [`scripts/visuals/`](scripts/visuals/) | Paper plotting scripts |
| [`assets/paper/`](assets/paper/) | Manuscript PDF and figure PDFs |
| [`assets/docs/setup-and-pipeline.md`](assets/docs/setup-and-pipeline.md) | Detailed execution guide |

## Citation

If you use this work, please cite the [preprint](assets/paper/paper.pdf):

```bibtex
@misc{walimbe2026tempochange,
  title = {Detecting Changes in Power Plant NOx Emissions Using Deep Learning and TEMPO Satellite Imagery},
  author = {Walimbe, Pranav and Murayama, Hikari and Cohen, Ron},
  year = {2026},
  note = {Non-peer-reviewed preprint},
  url = {https://github.com/pranav-walimbe/powerplant-modeling/blob/main/assets/paper/paper.pdf}
}
```

## License

This repository is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). See [LICENSE](LICENSE) for the full terms. Third-party materials remain subject to their own licenses and notices.
