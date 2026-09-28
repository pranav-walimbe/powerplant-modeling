# Active delta-model experiment plan

This file records the current plan for the next delta-modeling iteration. Keep
prior evidence, current defaults, proposed ablations, and open questions
separate. Run comparisons on identical geographic splits and timestep cohorts.
Do not infer a design win from overall accuracy alone.

## Goal and order of work

Build a model that predicts decrease, steady, and increase classes from a
sequence of TEMPO NO2 and weather rasters. The next experiments should identify
whether target construction, normalization, contextual conditioning, and an
explicit transport residual improve geographic generalization.

Run the work in this order:

1. Reproduce the current model on the new timestep contract.
2. Sweep AOI-selection and delta-class thresholds.
3. Compare normalization and context designs.
4. Add transport-derived rasters.
5. Test diffusion or decay only after advection adds value.

Avoid a full factorial sweep. Fix each upstream choice before testing the next
stage, then rerun the final design across multiple fixed seeds.

## Evidence from the previous timestep design

The earlier experiment log provides controls, not conclusions for the new
timestep setup:

- learned late fusion reached 0.6092 overall accuracy and 0.3120 decrease
  recall;
- independent seasonal and vision-only models reached 0.5991 overall accuracy,
  but decrease recall fell to 0.1151;
- post-ConvGRU FiLM reached 0.5909 overall accuracy and 0.3820 decrease recall;
- per-timestep context broadcast into the ConvGRU reached 0.6071 overall
  accuracy and 0.2597 decrease recall;
- removing solar hour reduced full-model accuracy by 0.0135, and removing day
  of year reduced it by 0.0262; and
- AOI-normalized learned fusion reached 0.6092, while the global-normalization
  run reached 0.6004.

These single-seed runs show that temporal context matters. They do not establish
late fusion as the best conditioning mechanism. FiLM placement changed the
class tradeoff and deserves a controlled test under the new timestep contract.

## Evaluation contract

Use geographically disjoint train, validation, and test AOIs. Fit every
normalization statistic on the training split. Tune thresholds, model choices,
and checkpoints with validation data and keep the test split frozen until the
design is fixed.

Report:

- overall accuracy, balanced accuracy, macro F1, and macro AUROC;
- per-class recall, with explicit attention to decrease recall;
- confusion matrices and predicted-class frequencies;
- performance by AOI, season, local solar hour, wind regime, timestep gap,
  plant scale, and target magnitude; and
- mean and dispersion across fixed seeds for the final comparisons.

Select models by validation macro F1 or balanced accuracy with a minimum
decrease-recall requirement. Overall accuracy can hide a collapse in the
decrease class.

## AOI and delta thresholds

Treat the AOI threshold and class-delta thresholds as data-definition choices.
Use a small grid rather than tuning them jointly with architecture choices. For
each setting, record:

- retained AOIs and records by split;
- class balance and target magnitude;
- AOI, season, solar-hour, wind, and plant-size coverage; and
- recall for large decreases and increases.

Choose thresholds on validation results and scientific usefulness. A threshold
that raises headline accuracy by removing difficult decreases does not improve
the original task.

## Normalization and context

### Current default

Keep AOI-specific robust NO2 normalization as the primary design. It removes
persistent site-level offsets and lets the raster encoder focus on plume shape
and evolution. Retain global weather normalization.

The initial context vector contains six values:

- local solar hour sine and cosine;
- day-of-year sine and cosine;
- `log1p(num_ng_units)`; and
- `log1p(num_coal_units)`.

Standardize the two transformed unit counts with training-split statistics.
Zeros retain the meaning that an AOI has no units of that fuel type.

Unit counts provide a coarse plant-scale proxy. They do not measure operating
state, and one large unit can exceed several small units in capacity. Compare
fuel-specific nameplate capacity later:

- `log1p(total_ng_capacity_mw)`; and
- `log1p(total_coal_capacity_mw)`.

Do not use historical average NO2 emissions in the first context experiment.
That feature could encode AOI identity or part of the prediction target.
Timestamp-aligned heat input, load, or operating-unit counts require a separate
leakage audit and must precede the prediction target.

### Conditioning experiments

Keep the existing learned late-fusion model as the required reference. Compare
it with:

1. vision only;
2. context only;
3. bounded, identity-initialized FiLM in the middle and deep frame-encoder
   blocks;
4. context conditioning inside the temporal module; and
5. spatial plus temporal conditioning.

Do not add a direct context-to-output path to the FiLM variants. Use a small
context encoder, bounded modulation, and context dropout. Include shuffled-
context evaluation to detect shortcuts.

### Normalization ablations

Compare the primary AOI-normalized model against:

- global NO2 normalization plus the same context vector;
- global NO2 normalization plus fuel-specific capacity; and
- AOI normalization with the AOI median and robust scale supplied as context.

The third option preserves stable within-AOI scaling while allowing the model
to recover absolute concentration scale when useful.

## Transport residual rasters

Use transport as a baseline forecast that the model can correct. For each
observed transition, define

\[
\widehat{X}^{adv}_t = A(X_{t-1}; u, v, \Delta t)
\]

and

\[
R_t = X_t - \widehat{X}^{adv}_t.
\]

A positive residual means that the current raster contains more NO2 than the
transport forecast; a negative residual means that it contains less. Interpret
the residual as a mixture of emissions, chemistry, dilution, deposition,
retrieval error, wind error, and geolocation error.

### Initial transport operator

- Work in physical NO2 units before classifier normalization.
- Average or interpolate winds over the interval between scans.
- Convert wind and elapsed time into east-west and north-south pixel
  displacements.
- Apply backward semi-Lagrangian warping with bilinear interpolation.
- Start with horizontal advection only.
- Preserve a support mask that measures valid source coverage for each warped
  pixel.

Do not replace the raw raster sequence. Add transported and residual views so
the network can reject a poor transport estimate. For each transition, test the
following inputs:

- current NO2;
- transported previous NO2;
- transport residual;
- east-west and north-south displacement;
- transport support or confidence; and
- the original NO2 validity information.

Never encode an invalid residual as an ordinary zero without a validity channel.

### Pixel uncertainty and smoothing

Confidence should decrease when trajectories leave the raster, source support
is missing, displacement is large, winds disagree across the interval, or wind
shear makes one horizontal field inadequate. A later uncertainty experiment
may perturb wind and timing inputs, then supply the ensemble residual mean and
variance.

Small registration errors can create adjacent positive and negative residuals.
Compare matched smoothing of the transported and current rasters before
subtraction:

- no smoothing;
- Gaussian smoothing with sigma 0.5 pixel; and
- Gaussian smoothing with sigma 1 pixel.

Keep the unsmoothed residual when adding a smoothed residual. Do not smooth only
the previous raster, and avoid filters large enough to erase plume structure on
a 24 by 24 grid.

### Transport ablation sequence

1. Raw raster sequence.
2. Ordinary difference, `current - previous`.
3. Transported previous raster.
4. Advection residual.
5. Advection residual plus support or confidence.
6. Sharp and smoothed residuals together.
7. Wind-perturbation ensemble statistics.

Evaluate each variant by timestep gap, wind speed and direction, displacement,
boundary-support fraction, AOI, and target class. If advection does not beat the
ordinary difference, inspect wind height, timing, grid conversion, and
geolocation before adding more physics.

## Diffusion, chemistry, and decay

Do not include prescribed decay in the first transport residual. Satellite NO2
studies find that an effective lifetime also absorbs dispersion and deposition,
and that lifetime and plume width can compensate for each other. Single coarse
images seldom constrain both.

After advection proves useful, test these additions one at a time:

1. a small fixed or fitted diffusion term;
2. exponential decay,
   `exp(-delta_t / tau) * advected_previous`; and
3. decay conditioned on a small set of justified variables such as solar hour
   or temperature.

Fit diffusion and lifetime values on training data and select them with
validation data. Keep the learned transport residual even when adding decay so
the model can represent emissions and misspecified chemistry.

## Open questions

- Should fuel-specific nameplate capacity replace unit counts after the first
  context run?
- Do AOI normalization median and scale recover useful absolute information
  without the instability of global NO2 normalization?
- Which model layer benefits from context under the new timestep contract?
- Does the wind product represent plume-level transport well enough at this
  grid scale?
- Does matched smoothing improve transport residuals without removing plume
  edges?
- Do diffusion or decay improve validation performance after accounting for
  transport uncertainty?

## Research basis

- [NowcastNet](https://www.nature.com/articles/s41586-023-06184-4) separates
  backward semi-Lagrangian advection from a learned source-sink residual.
- [LUPIN](https://doi.org/10.1016/j.acags.2025.100296) transforms precipitation
  into Lagrangian coordinates and learns changes after motion compensation.
- [Jin et al.](https://amt.copernicus.org/articles/14/7929/2021/) show that NO2
  lifetime and plume spread are difficult to identify from single satellite
  overpasses and can compensate for one another.
- [Beirle and Wagner](https://amt.copernicus.org/articles/17/3439/2024/) discuss
  wind-field uncertainty and effective lifetime estimation from aggregated
  satellite NO2 plumes.
