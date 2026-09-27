# Modeling results

## Experiment log

Class accuracy is recall within each true class.

| Experiment description | Run ID | Overall test accuracy | Decrease accuracy | Steady accuracy | Increase accuracy |
|---|---|---:|---:|---:|---:|
| Baseline with AOI-specific NO2 normalization and 64-channel vision model | `delta_category_classification_20260927_042746` | 0.5983 | 0.2806 | 0.3582 | 0.8850 |
| Capacity step-down from 64 to 48 vision channels | `delta_category_classification_20260927_061047` | 0.5887 | 0.3180 | 0.3100 | 0.8903 |
| Soft cross-entropy with AOI-threshold targets and tau 0.05 on the 48-channel model | `delta_category_classification_20260927_065341` | 0.5863 | 0.3426 | 0.3326 | 0.8600 |
| Independent seasonal and vision-only models with hard CE and 48 vision channels | `delta_category_classification_20260927_071252` | 0.5991 | 0.1151 | 0.3577 | 0.9392 |

## Baseline run details

Run: `delta_category_classification_20260927_042746`

### Run summary

| Item | Value |
|---|---|
| Seed | 42 |
| Train / validation / test records | 74,241 / 15,164 / 16,417 |
| Train / validation / test AOIs | 351 / 89 / 64 |
| Target classes | Decrease / steady / increase |
| Seasonal model | 291-parameter MLP |
| Vision-seasonal model | 477,923-parameter ConvGRU fusion model |
| NO2 completion | Masked-model checkpoint `masked_no2_20260924_183700` |
| NO2 normalization | AOI-specific median and `IQR / 1.349` |
| Weather normalization | Global masked-model statistics |
| Selected seasonal epoch | 16 |
| Selected vision-seasonal epoch | 2 |

### Split metrics

| Split | Model | Accuracy | Balanced accuracy | Macro F1 | Macro AUROC |
|---|---|---:|---:|---:|---:|
| Train | Seasonal | 0.5377 | 0.5103 | 0.4929 | 0.7056 |
| Train | Vision-seasonal | 0.5725 | 0.5451 | 0.5473 | 0.7470 |
| Validation | Seasonal | 0.5555 | 0.5360 | 0.5164 | 0.7172 |
| Validation | Vision-seasonal | 0.5824 | 0.5655 | 0.5696 | 0.7383 |
| Test | Seasonal | 0.5331 | 0.4909 | 0.4504 | 0.6923 |
| Test | Vision-seasonal | 0.5983 | 0.5079 | 0.5036 | 0.7433 |

### Per-class recall

| Split | Model | Decrease | Steady | Increase |
|---|---|---:|---:|---:|
| Train | Seasonal | 0.5342 | 0.2126 | 0.7841 |
| Train | Vision-seasonal | 0.4079 | 0.4570 | 0.7706 |
| Validation | Seasonal | 0.6181 | 0.2280 | 0.7618 |
| Validation | Vision-seasonal | 0.5330 | 0.4376 | 0.7260 |
| Test | Seasonal | 0.4601 | 0.1915 | 0.8209 |
| Test | Vision-seasonal | 0.2806 | 0.3582 | 0.8850 |

### Test accuracy by AOI characteristic

| Characteristic | Stratum | AOIs | Records | Seasonal | Vision-seasonal | Difference |
|---|---|---:|---:|---:|---:|---:|
| Average heat input | Low | 21 | 5,488 | 0.5375 | 0.6039 | +0.0663 |
| Average heat input | Middle | 22 | 5,916 | 0.5546 | 0.6077 | +0.0531 |
| Average heat input | High | 21 | 5,013 | 0.5029 | 0.5813 | +0.0784 |
| Total unit count | Low | 21 | 3,199 | 0.2488 | 0.3951 | +0.1463 |
| Total unit count | Middle | 22 | 6,907 | 0.5826 | 0.6428 | +0.0602 |
| Total unit count | High | 21 | 6,311 | 0.6230 | 0.6527 | +0.0296 |
