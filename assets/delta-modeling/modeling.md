# Modeling

The delta trainer compares two classifiers on the same geographic splits and
fixed seed. Separable bilinear interpolation fills low-quality NO2 pixels before
classifier training. The classifier also receives the original NO2 quality mask
so it can distinguish observations from filled values.

## Contract

| Component | Choice |
|---|---|
| Target | Stored `delta_category` |
| Classes | `decrease`, `steady`, `increase` |
| Observation sequence | Five frames at `24 x 24` pixels |
| Observation channels | Filled NO2, NO2 quality, wind-u, wind-v, temperature |
| Transport sequence | Four advection residuals and residual-quality rasters |
| Seasonal context | Midpoint day of year and local solar hour |
| Loss | Training-prevalence-weighted class cross-entropy |
| Selection | Highest validation macro F1; weighted validation loss breaks ties |
| Primary result | Context-aware vision classifier |

Both models use context at the midpoint between the first and fifth TEMPO
scans. The pipeline represents day of year and local solar hour with sine and
cosine pairs, then normalizes those four features with training-split
statistics.

## Raster completion and quality

The classifier pipeline fits a global mean and population standard deviation to
every raster channel using only the training split. It then fills missing NO2
with the mean of horizontal and vertical linear interpolation in that frozen
normalized space. Rows or columns without visible support fall back to the
frame's visible-pixel mean.

Training stores five channels in job-local memory-mapped arrays: four normalized
physical inputs plus the original NO2 mask. Observed pixels receive quality 1;
filled pixels receive quality 0. The classifier processes every filled value and
uses quality as a separate input channel.

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
- `vision`: the context-aware vision classifier.

The observation branch processes five frames, while the residual branch
processes four transitions without creating a synthetic first residual. Each
later observation embedding is fused with its aligned residual embedding by a
`1 x 1` projection before the shared ConvGRU. The classifier pools the final
recurrent map. A small MLP encodes the sample's midpoint context once, and its
embedding is concatenated with the pooled visual embedding immediately before
classification. The model width is 64.

Training drops the complete midpoint context embedding for a random share of
samples. The job also saves predictions and metrics from shuffled-context
inference.

## Optimization

All classifiers use inverse-frequency class weights computed only from the
training split. Each class therefore contributes equal total weight to the
cross-entropy objective.

Validation macro F1 controls checkpointing and early stopping.
Training-weighted validation cross-entropy breaks macro-F1 ties and drives the
learning-rate scheduler. The maintained launcher uses batch size 128, model
width 64, head width 48, dropout 0.20, learning rate
`6.0378112892784365e-5`, weight decay `1e-4`, early-stopping patience 10, and
seed 42.

Run the production workflow with:

```bash
sbatch scripts/slurm/train_model.sh
```

Training always fills missing NO2 pixels with separable bilinear interpolation
in the classifier's training-split normalization. There is no fill-strategy
CLI option or masked-model checkpoint dependency.

## Hyperparameter optimization

`modeling.optimize` runs a seeded TPE search with median pruning.
It reuses train-only normalization and completed train/validation rasters across
trials. It never loads or evaluates the test split. The search varies learning
rate, weight decay, dropout, and model width.
Learning rate is sampled log-uniformly from `1e-5` through `1e-4`. The
categorical choices are weight decay `{1e-4, 1e-3}`, dropout `{0.2, 0.4}`,
and model width `{32, 64, 96}`.

Run the maintained 24-trial Savio search with:

```bash
sbatch scripts/slurm/optimize_model.sh
```

Pass `--study-dir` to resume an existing SQLite-backed study. The study writes
one validation-only summary per trial, `trials.csv`, and `study_summary.json`.

## Evaluation and artifacts

`results.json` reports overall accuracy, balanced accuracy, macro F1, macro
one-vs-rest AUROC, per-class accuracy (recall), per-class F1,
confusion matrices, and predicted-class frequencies.
The run also writes row-level predictions, best checkpoints, normalization
state, residual statistics, and shuffled-context diagnostics.

The trainer produces five figures:

- `split_class_f1.png` compares macro F1 and per-class F1 with class support for
  every split;
- `split_class_accuracy.png` compares overall and per-class accuracy with class
  support for every split;
- `training_curves.png` shows train and validation loss;
- `performance_by_characteristic.png` compares test macro F1 by AOI heat input
  and unit count, plus test accuracy by calendar month and midpoint local solar
  hour;
- `joint_prediction_samples.png` shows 20 deterministic, class-balanced test
  examples ordered from strongest NOx decrease to strongest increase. It includes
  all five NO2 inputs, four transport residuals, the true class, and both true
  and predicted contribution channels.

The result email attaches all four figures. Test plots describe fixed
checkpoints and do not participate in model selection.
