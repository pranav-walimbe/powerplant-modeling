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

| Ablation | Run ID | Split | Class | Seasonal F1 | Vision F1 | Seasonal change | Vision change |
|---|---|---|---|---:|---:|---:|---:|

## Experiment log

| Design tested | Run ID | Seasonal test macro F1 | Vision test macro F1 |
|---|---|---:|---:|
| Freeze the trained seasonal model and train the unchanged vision model to predict additive corrections to its logits | `delta_logit_correction_20261001_063646` | 0.4938 | 0.5351 |
| Remove additive FiLM shifts so seasonal context can only rescale each branch's per-timestep features | `delta_scale_only_film_20261001_070955` | 0.4938 | 0.5424 |
| Remove FiLM and concatenate the midpoint seasonal embedding with the temporally aggregated visual embedding | `delta_no_film_20261001_074126` | 0.4938 | 0.5416 |
| Increase the baseline FiLM vision model width from 48 to 64 while keeping the classifier head width at 48 | `delta_model_width_64_20261001_080816` | 0.4938 | 0.5514 |
