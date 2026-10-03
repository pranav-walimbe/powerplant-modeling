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
| Loss | Weighted class cross-entropy plus two-channel unbalanced Sinkhorn divergence |
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
- `vision`: the joint context-aware class and contribution-raster predictor.

The observation branch processes five frames. The residual branch processes
four transitions. A shared MLP encodes solar-hour sine/cosine and day-of-year
sine/cosine at each observation timestamp. Separate zero-initialized heads
produce bounded channel-wise scale and shift values for the observation and
residual spatial bottlenecks. The first observation is modulated with its own
context and initializes the temporal state without a synthetic residual. Each
later observation and aligned residual are modulated with that timestep's
context, fused with a `1 x 1` projection, and passed into the shared ConvGRU.
The classifier pools the final recurrent map. A separate decoder upsamples the
same map into bounded `24 x 24` decrease and increase contribution rasters.
The model width is 64.

This is identity-initialized FiLM with unbounded scale and additive shift.
Training drops the full context sequence for a random share of samples. The
job also saves predictions and metrics from shuffled-context inference.

## Spatial contribution targets

Before training, the pipeline parses `t0_source_pixel_nox` through
`t4_source_pixel_nox` and aligns facilities by ID. A facility omitted at a
timestep has zero emissions. It computes the facility's early and late
elapsed-time-weighted trapezoidal means with the same intervals used by the
class label, then subtracts early from late.

For an increase sample, positive facility changes are normalized to sum to one
and Gaussian-splatted into the increase channel. For a decrease sample, the
magnitudes of negative facility changes are normalized and splatted into the
decrease channel. The other channel is zero. Both channels are zero for steady
samples. Thus these are class-conditioned attribution targets: a zero steady
target does not assert that every facility was unchanged, because opposing
facility changes may still cancel within the steady threshold.

Source coordinates retain their fractional positions on the `24 x 24` input
grid. Each clipped Gaussian is renormalized, so the active channel retains unit
mass even near an AOI edge.

## Optimization

All classifiers use inverse-frequency class weights computed only from the
training split. Each class therefore contributes equal total weight to the
cross-entropy objective. The joint model compares each predicted contribution
channel with its target using a debiased unbalanced Sinkhorn divergence. This
penalizes spatially distant mass more than nearby mass, while mass relaxation
keeps all-zero steady and inactive channels valid. The two channel losses are
averaged per sample. Its total loss is
`CE + raster_loss_weight * sinkhorn_loss`; `RASTER_LOSS_WEIGHT` configures this
weight and defaults to 1.0.

Sinkhorn scaling uses a separable Gaussian kernel in float32, avoiding a dense
pairwise pixel-cost matrix. `SINKHORN_BLUR_CELLS`, `SINKHORN_REACH_CELLS`, and
`SINKHORN_ITERATIONS` default to 1.5, 6.0, and 30. The output head starts near
one unit of total mass across both channels instead of the sigmoid default of
0.5 per pixel. `CONTRIBUTION_SPLAT_SIGMA` controls target Gaussian width in
`24 x 24` cells and defaults to 1.5. Set it to zero for nearest-cell targets.

Validation macro F1 controls checkpointing and early stopping.
Training-weighted validation cross-entropy breaks macro-F1 ties and drives the
learning-rate scheduler. The maintained launcher uses batch size 128, model
width 64, head width 48, dropout 0.30, and seed 42. These rules are fixed
without a parameter-selection run.

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

The trainer produces four figures:

- `split_class_f1.png` compares macro F1 and per-class F1 with class support for
  every split;
- `training_curves.png` shows train and validation loss;
- `accuracy_by_characteristic.png` compares test accuracy by AOI heat input,
  unit count, calendar month, and midpoint local solar hour.
- `joint_prediction_samples.png` shows 20 deterministic, class-balanced test
  examples ordered from strongest NOx decrease to strongest increase. It includes
  all five NO2 inputs, four transport residuals, the true class, and both true
  and predicted contribution channels.

The result email attaches all four figures. Test plots describe fixed
checkpoints and do not participate in model selection.
