# Modeling

The delta trainer compares four classifiers on the same geographic splits and
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
| Sample context | Midpoint solar-hour sine/cosine and day-of-year sine/cosine |
| Loss | Unweighted three-class cross-entropy |
| Selection | Lowest validation cross-entropy for each model |
| Primary result | Context-conditioned two-branch transport classifier |

The loader computes context at the midpoint between the first and fifth TEMPO
scans. It converts UTC to local solar time with longitude and normalizes the
four cyclic features with training-split statistics.

## Raster completion and quality

The masked model uses its checkpoint's global statistics to fill NO2. Training
stores five channels in job-local memory-mapped arrays: the four completed
physical inputs plus the original NO2 mask. Observed pixels receive quality 1;
filled pixels receive quality 0. The classifier processes every filled value
and uses quality as a separate input channel.

The classifier applies AOI-specific median and robust scale to NO2. It uses the
masked checkpoint's global mean and standard deviation for temperature and
wind. Training fits all normalization state on the training split.

## Transport residuals

The GPU transport operator converts normalized NO2 back to physical units. For
each of four intervals, it averages endpoint 80-meter winds and performs
backward semi-Lagrangian warping with bilinear interpolation. It subtracts the
transported previous observation from the current observation.

Residual quality equals current NO2 quality multiplied by warped source NO2
quality. Backtraces outside the raster receive zero quality while the residual
value remains available to the encoder. A training-only pass fits
quality-weighted residual mean and standard deviation.

The operator accepts an optional decay lifetime and optional interval wind
fields. These hooks support later decay and elevated-wind or plume-rise
experiments. The horizontal-advection baseline leaves both disabled.

## Models

The training job fits these models:

- `seasonal`: the context-only MLP diagnostic;
- `learned_fusion`: the existing late-fusion reference;
- `transport_image_only`: the two raster branches without context;
- `transport_context`: the same two branches with bottleneck conditioning.

The observation branch processes five frames. The residual branch processes
four transitions. Each branch uses a spatial encoder followed by its own
ConvGRU. The classifier concatenates both temporal embeddings.

The context model sends the four-value vector through a small MLP. Separate
projection heads produce bounded scale changes for the two branch bottlenecks.
The projections start at identity, use no additive shift, and share one sample
context across all timesteps. Context has no direct path to the classifier.
Training drops context for a random share of samples. The job also saves
predictions and metrics from shuffled-context inference.

## Optimization

All classifiers use AdamW, gradient clipping, mixed precision, validation-loss
scheduling, and early stopping. The maintained launcher uses batch size 128,
model width 48, head width 48, dropout 0.30, and seed 42. It does not tune new
transport or conditioning parameters.

Run the production workflow with:

```bash
sbatch scripts/slurm/train_model.sh
```

## Evaluation and artifacts

`results.json` reports accuracy, balanced accuracy, macro F1, macro one-vs-rest
AUROC, per-class recall, confusion matrices, and predicted-class frequencies.
The run also writes row-level predictions, best checkpoints, normalization
state, residual statistics, and shuffled-context diagnostics.

The trainer produces three figures:

- `split_class_accuracy.png` compares overall and per-class performance;
- `training_curves.png` shows train and validation loss;
- `accuracy_by_characteristic.png` compares test accuracy by AOI heat input,
  unit count, calendar month, and midpoint local solar hour.

The test plots describe the fixed model and do not participate in model
selection.
