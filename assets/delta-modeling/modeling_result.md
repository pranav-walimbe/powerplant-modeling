# Modeling results

## Base run

Run: `delta_category_classification_20261001_053159`

Records are the number of true examples for each class in the split.

| Split | Class | Records | Seasonal F1 | Vision F1 |
|---|---|---:|---:|---:|
| Train | Decrease | 17,130 | 0.5457 | 0.6011 |
| Train | Steady | 23,289 | 0.3613 | 0.5521 |
| Train | Increase | 29,888 | 0.6750 | 0.6739 |
| Validation | Decrease | 3,198 | 0.4356 | 0.5406 |
| Validation | Steady | 8,589 | 0.3885 | 0.5992 |
| Validation | Increase | 10,411 | 0.6346 | 0.6506 |
| Test | Decrease | 3,184 | 0.3975 | 0.4354 |
| Test | Steady | 7,764 | 0.3817 | 0.5400 |
| Test | Increase | 10,093 | 0.7022 | 0.6335 |

## Ablation table

Class F1 triplets use decrease / steady / increase order.

| Ablation | Run ID | Train seasonal | Train vision | Validation seasonal | Validation vision | Test seasonal | Test vision | Test vision macro F1 |
|---|---|---|---|---|---|---|---|---:|
| No temperature | `delta_ablation_no_temperature_20261001_224152` | 0.5457 / 0.3612 / 0.6750 | 0.6010 / 0.4479 / 0.7136 | 0.4357 / 0.3884 / 0.6346 | 0.4880 / 0.5191 / 0.7157 | 0.3972 / 0.3819 / 0.7022 | 0.4400 / 0.4488 / 0.7107 | 0.5331 |
| No wind | `delta_ablation_no_wind_20261002_001550` | 0.5457 / 0.3612 / 0.6750 | 0.6105 / 0.5409 / 0.7030 | 0.4357 / 0.3884 / 0.6346 | 0.5342 / 0.6024 / 0.7005 | 0.3972 / 0.3819 / 0.7022 | 0.4329 / 0.5206 / 0.6693 | 0.5410 |
| No transport | `delta_ablation_no_transport_20261002_002527` | 0.5457 / 0.3612 / 0.6750 | 0.6333 / 0.5574 / 0.6732 | 0.4357 / 0.3884 / 0.6346 | 0.5127 / 0.5853 / 0.6341 | 0.3972 / 0.3819 / 0.7022 | 0.4310 / 0.5009 / 0.6076 | 0.5132 |
| No class weights | `delta_ablation_no_loss_weights_20261002_003352` | 0.5053 / 0.3712 / 0.6852 | 0.6151 / 0.5883 / 0.7245 | 0.4374 / 0.4108 / 0.6544 | 0.5154 / 0.6123 / 0.6841 | 0.3750 / 0.3932 / 0.7039 | 0.4284 / 0.5511 / 0.6826 | 0.5540 |
| No NO2 | `delta_ablation_no_no2_20261002_004354` | 0.5457 / 0.3612 / 0.6750 | 0.5950 / 0.5210 / 0.6865 | 0.4357 / 0.3884 / 0.6346 | 0.5351 / 0.5839 / 0.6675 | 0.3972 / 0.3819 / 0.7022 | 0.4314 / 0.5119 / 0.6597 | 0.5343 |

## Experiment log

| Design tested | Run ID | Seasonal test macro F1 | Vision test macro F1 | Vision test decrease F1 | Vision test steady F1 | Vision test increase F1 |
|---|---|---:|---:|---:|---:|---:|
| Freeze the trained seasonal model and train the unchanged vision model to predict additive corrections to its logits | `delta_logit_correction_20261001_063646` | 0.4938 | 0.5351 | 0.4402 | 0.5190 | 0.6461 |
| Remove additive FiLM shifts so seasonal context can only rescale each branch's per-timestep features | `delta_scale_only_film_20261001_070955` | 0.4938 | 0.5424 | 0.4476 | 0.4871 | 0.6926 |
| Remove FiLM and concatenate the midpoint seasonal embedding with the temporally aggregated visual embedding | `delta_no_film_20261001_074126` | 0.4938 | 0.5416 | 0.4200 | 0.5294 | 0.6755 |
| Increase the baseline FiLM vision model width from 48 to 64 while keeping the classifier head width at 48 | `delta_model_width_64_20261001_080816` | 0.4938 | 0.5514 | 0.4651 | 0.5109 | 0.6782 |
| Restore the baseline FiLM design and replace hard-label cross-entropy with threshold-aware soft cross-entropy at a 50 lb/hr temperature | `delta_soft_cross_entropy_20261001_183615` | 0.5122 | 0.5481 | 0.4391 | 0.5462 | 0.6590 |
| Add a 12x12 smoothed delta-raster decoder to the baseline FiLM model and jointly optimize unit-weight normalized raster L1 with hard cross-entropy | `delta_joint_raster_12x12_20261001_193944` | 0.4938 | 0.5469 | 0.4180 | 0.5277 | 0.6949 |
| Restore the baseline FiLM model and smooth each physical transport residual with a quality-weighted normalized Gaussian at sigma 1 native pixel | `delta_quality_aware_residual_smoothing_20261001_200841` | 0.4938 | 0.5448 | 0.4478 | 0.4942 | 0.6926 |
