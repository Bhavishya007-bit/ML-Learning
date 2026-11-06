# Evaluation report

Data source: **synthetic_demo**

Synthetic data results are software demonstrations, not insurance benchmarks.

Rows: 5,000; features: 130; split: 60%/20%/20%.

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| median | 2497.90 | 4494.42 | -0.091 |
| linear_regression | 1818.19 | 3312.92 | 0.407 |
| xgboost | 1223.51 | 2204.85 | 0.737 |

Measured MAE reduction versus linear: 32.71%.

Both regressions use log1p(loss); all metrics above use original loss units.
See synthetic_metrics.json for parameters, versions, validation scores and bootstrap uncertainty.
