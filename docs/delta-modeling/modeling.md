# Modeling

The delta trainer compares two classifiers on the same geographic splits and
fixed seed. A pretrained masked model fills low-quality NO2 pixels before
classifier training. The classifier still receives the original NO2 quality
mask, so it can distinguish observations from filled values.

## Contract

| Component | Choice |
|---|---|
| Target | Stored `delta_category` |
| Classes | `decrease`, `steady`, `increase` |
| Observation sequence | Five frames at `24 x 24` pixels |
| Observation channels | Filled NO2, NO2 quality, wind-u, wind-v, temperature |
| Transport sequence | Four advection residuals and residual-quality rasters |
| Seasonal context | Midpoint solar hour for the baseline; per-timestep solar hour for vision |
| Loss | Training-prevalence-weighted three-class cross-entropy |
| Selection | Highest validation macro F1; weighted validation loss breaks ties |
| Primary result | Context-aware vision classifier |

The seasonal baseline uses context at the midpoint between the first and fifth
TEMPO scans. The vision model uses context at all five observation timestamps.
It converts UTC to local solar time with longitude, rounds vision-model hours
to the nearest whole hour, and normalizes the four cyclic features with
training-split statistics.

## Raster completion and quality

The masked model uses its checkpoint's global statistics to fill NO2. Training
stores five channels in job-local memory-mapped arrays: four inputs in the
masked model's normalized space plus the original NO2 mask. Observed pixels
receive quality 1; filled pixels receive quality 0. The classifier processes
every filled value and uses quality as a separate input channel.

The masked model's checkpoint statistics are used only during raster
completion. The pipeline fits a global median and robust scale to
training-split NO2. It also fits global means and standard deviations to
training-split temperature and wind. The completed train, validation, and test
rasters are converted from the masked model's normalized space into this
frozen classifier normalization.

## Transport residuals

The data-loader's CPU transport transform converts normalized NO2 back to
physical units. For each of four intervals, it averages endpoint 80-meter
winds and performs backward semi-Lagrangian warping with bilinear
interpolation. It subtracts the transported previous observation from the
current observation before the batch is transferred to the GPU.

Residual quality equals current NO2 quality multiplied by warped source NO2
quality. Backtraces outside the raster receive zero quality while the residual
value remains available to the encoder. A training-only pass fits
quality-weighted residual mean and standard deviation.

The maintained configuration blends endpoint winds equally, scales wind speed
by 0.8, disperses the transported enhancement with a one-hour width of one
pixel, and applies an eight-hour effective decay lifetime. The transform can
accept externally supplied interval winds, but the pipeline does not implement
elevated-wind selection or plume rise. The run and checkpoint metadata record
the active transport values directly from the transport configuration.

## Models

The training job fits these models:

- `seasonal`: the context-only MLP diagnostic;
- `vision`: the context-aware observation and transport-residual classifier.

The observation branch processes five frames. The residual branch processes
four transitions. A shared MLP encodes solar-hour sine/cosine and day-of-year
sine/cosine at each observation timestamp. Separate zero-initialized heads
produce bounded channel-wise scale and shift values for the observation and
residual spatial bottlenecks. The first observation is modulated with its own
context and initializes the temporal state without a synthetic residual. Each
later observation and aligned residual are modulated with that timestep's
context, fused with a `1 x 1` projection, and passed into the shared ConvGRU.

This is identity-initialized FiLM with unbounded scale and additive shift.
Training drops the full context sequence for a random share of samples. The
job also saves predictions and metrics from shuffled-context inference.

## Optimization

All classifiers use inverse-frequency class weights computed only from the
training split. Each class therefore contributes equal total weight to the
cross-entropy objective. Validation macro F1 controls checkpointing and early
stopping. Training-weighted validation cross-entropy breaks macro-F1 ties and
drives the learning-rate scheduler. The maintained launcher uses batch size
128, model width 48, head width 48, dropout 0.30, and seed 42. These rules are
fixed without a parameter-selection run.

Run the production workflow with:

```bash
sbatch scripts/slurm/train_model.sh
```

## Evaluation and artifacts

`results.json` reports accuracy, balanced accuracy, macro F1, macro one-vs-rest
AUROC, per-class F1, per-class recall, confusion matrices, and predicted-class
frequencies.
The run also writes row-level predictions, best checkpoints, normalization
state, residual statistics, and shuffled-context diagnostics.

The trainer produces three figures:

- `split_class_f1.png` compares macro F1 and per-class F1 with class support for
  every split;
- `training_curves.png` shows train and validation loss;
- `accuracy_by_characteristic.png` compares test accuracy by AOI heat input,
  unit count, calendar month, and midpoint local solar hour.

The test plots describe the fixed model and do not participate in model
selection.
