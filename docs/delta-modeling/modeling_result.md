# Modeling results

## Base run

Run: `delta_category_classification_20261001_053159`

Class accuracy is recall within each true class. Records are the number of
true examples for that class in the split.

| Split | Model | Class | Records | Class accuracy |
|---|---|---|---:|---:|
| Train | Seasonal | Decrease | 17,130 | 0.5480 |
| Train | Seasonal | Steady | 23,289 | 0.3120 |
| Train | Seasonal | Increase | 29,888 | 0.7451 |
| Train | Vision | Decrease | 17,130 | 0.5568 |
| Train | Vision | Steady | 23,289 | 0.6392 |
| Train | Vision | Increase | 29,888 | 0.6195 |
| Validation | Seasonal | Decrease | 3,198 | 0.5635 |
| Validation | Seasonal | Steady | 8,589 | 0.3319 |
| Validation | Seasonal | Increase | 10,411 | 0.6535 |
| Validation | Vision | Decrease | 3,198 | 0.5585 |
| Validation | Vision | Steady | 8,589 | 0.6711 |
| Validation | Vision | Increase | 10,411 | 0.5797 |
| Test | Seasonal | Decrease | 3,184 | 0.4570 |
| Test | Seasonal | Steady | 7,764 | 0.3096 |
| Test | Seasonal | Increase | 10,093 | 0.7710 |
| Test | Vision | Decrease | 3,184 | 0.4438 |
| Test | Vision | Steady | 7,764 | 0.6289 |
| Test | Vision | Increase | 10,093 | 0.5495 |

## Ablation table

| Ablation | Run ID | Split | Model | Class | Class accuracy | Change from base |
|---|---|---|---|---|---:|---:|

## Experiment log

| Experiment | Run ID | Status | Notes |
|---|---|---|---|
