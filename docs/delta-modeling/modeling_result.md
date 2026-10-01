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

| Experiment | Run ID | Status | Notes |
|---|---|---|---|
