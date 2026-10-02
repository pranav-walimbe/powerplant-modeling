# Modeling results

## Run summary

| Item | Value |
|---|---|
| Run directory | `masked_no2_20260924_183700` |
| Model | 394,209-parameter masked NO2 convolutional autoencoder |
| Data | 500,000 train, 50,000 validation, and 50,000 test scenes |
| Best checkpoint | Epoch 71 |
| Best validation normalized L1 | 0.33421 |
| Evaluation job | `39190726` |
| Evaluated test pixels | 2,305,623 artificially hidden valid pixels |
| Comparison basis | Same 50,000-record test split and synthetic masks |

## Test comparison with bilinear interpolation

| Test metric on hidden pixels | Epoch-71 model | Bilinear interpolation | Relative reduction |
|---|---:|---:|---:|
| Normalized MAE / L1 | 0.33602 | 0.40442 | 16.91% |
| Normalized RMSE | 0.47227 | 0.55159 | 14.38% |
