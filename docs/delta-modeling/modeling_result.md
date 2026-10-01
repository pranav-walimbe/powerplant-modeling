# Modeling results

## Base run

Run: `delta_category_classification_20261001_053159`

Records are the number of true examples for each class in the split.

| Split | Model | Class | Records | Class F1 |
|---|---|---|---:|---:|
| Train | Seasonal | Decrease | 17,130 | 0.5457 |
| Train | Seasonal | Steady | 23,289 | 0.3613 |
| Train | Seasonal | Increase | 29,888 | 0.6750 |
| Train | Vision | Decrease | 17,130 | 0.6011 |
| Train | Vision | Steady | 23,289 | 0.5521 |
| Train | Vision | Increase | 29,888 | 0.6739 |
| Validation | Seasonal | Decrease | 3,198 | 0.4356 |
| Validation | Seasonal | Steady | 8,589 | 0.3885 |
| Validation | Seasonal | Increase | 10,411 | 0.6346 |
| Validation | Vision | Decrease | 3,198 | 0.5406 |
| Validation | Vision | Steady | 8,589 | 0.5992 |
| Validation | Vision | Increase | 10,411 | 0.6506 |
| Test | Seasonal | Decrease | 3,184 | 0.3975 |
| Test | Seasonal | Steady | 7,764 | 0.3817 |
| Test | Seasonal | Increase | 10,093 | 0.7022 |
| Test | Vision | Decrease | 3,184 | 0.4354 |
| Test | Vision | Steady | 7,764 | 0.5400 |
| Test | Vision | Increase | 10,093 | 0.6335 |

## Ablation table

| Ablation | Run ID | Split | Model | Class | Class F1 | Change from base |
|---|---|---|---|---|---:|---:|

## Experiment log

| Experiment | Run ID | Status | Notes |
|---|---|---|---|
