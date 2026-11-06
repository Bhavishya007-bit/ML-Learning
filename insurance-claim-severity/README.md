# Insurance Claim Severity Prediction & Risk Segmentation

A reproducible Python + SQL project that predicts claim cost, compares linear regression with XGBoost, and groups claims into predicted severity bands.

**Status:** implementation and synthetic demonstration available. The real Allstate benchmark has not been run. No 23% real-data improvement is claimed. This implementation was created with AI assistance in November 2026; its commit history is not evidence of a September project.

## Quick start

Python 3.11 or 3.12 recommended. From this directory:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m claims demo --rows 5000 --output outputs/demo
```

The demo generates **artificial records with 130 inputs**. It tests the workflow, not performance on real insurance claims. It writes SQL data, quality checks, fitted model, metrics, split IDs, predictions, and segment summaries. Use a new output directory for each experiment.

## Real dataset

Obtain `train.csv` yourself from the official [Allstate Claims Severity competition](https://www.kaggle.com/competitions/allstate-claims-severity/data), following its access requirements and competition rules. Dataset files are not redistributed here.

The expected format is `id`, `cat1` through `cat116`, `cont1` through `cont14`, and `loss`. That is **116 categorical + 14 numerical = 130 predictors**, excluding ID and target. The published training set has 188,318 rows; this implementation validates the actual input and records its real row count rather than assuming that size. See the [Stanford project report](https://cs229.stanford.edu/proj2017/final-reports/5242012.pdf) for the published dataset dimensions. Anonymized variables do not provide a basis for inventing age, income, or medical meanings.

```bash
python -m claims train --csv data/train.csv --source allstate_kaggle_train --output outputs/allstate
python -m claims predict --model outputs/allstate/model.joblib --csv data/test.csv --output outputs/predictions.csv
```

Only load model files you created or trust: joblib uses pickle. Data, databases, models, and prediction files are ignored by Git.

## How it works

1. **Validate and ingest:** enforce required columns, positive unique IDs, numeric types, finite nonnegative target, and duplicate-record checks. Missing predictors are recorded and preserved. Missing target values are rejected.
2. **Relational storage:** SQLite tables hold ingestion provenance, claim outcomes, and feature records. Composite primary/foreign keys connect them. Parameterized inserts and a transaction prevent partial batches. SHA-256 deduplicates repeated file ingestion. SQLite makes the SQL workflow runnable without a database server.
3. **Split before fitting:** fixed random seed, 60% training / 20% validation / 20% test. IDs never become predictors. The fitted transformations never see validation or test statistics.
4. **Preprocess:** a Scikit-Learn `ColumnTransformer` learns category frequencies on training rows. Unknown categories map to zero; missing categories get their own frequency. Numerical columns use training medians and a quantile-to-normal transformation.
5. **Predict severity:** both linear regression and XGBoost fit `log1p(loss)` through `TransformedTargetRegressor`. Predictions use `expm1` and a nonnegative floor. A constant-median baseline is also reported.
6. **Select and evaluate:** compare two predeclared XGBoost depths using validation MAE in original loss units. Freeze the chosen model, evaluate the test set, and report MAE, RMSE, R² and a paired-bootstrap interval for the MAE difference. No refit on validation or test rows.
7. **Segment:** validation prediction tertiles define low/medium/high severity thresholds. Apply frozen thresholds to test/new predictions. Report actual costs and errors by band for inspection.

“Risk” here means a **predicted claim-severity band conditional on a claim**. It is not claim frequency, fraud probability, an underwriting decision, or total expected policy loss. Tertiles are an interpretable starting point, not calibrated business thresholds.

## Files to read

| File | Purpose |
|---|---|
| `claims/data.py` | Data contract, SQL ingestion and synthetic generator |
| `sql/schema.sql` | Relational keys and constraints |
| `sql/analysis.sql` | Example SQL analyses |
| `claims/features.py` | Frequency encoder and ColumnTransformer |
| `claims/model.py` | Baselines, model selection, metrics and segmentation |
| `claims/__main__.py` | Training, demo and prediction commands |
| `tests/test_pipeline.py` | Validation, leakage, SQL and end-to-end checks |
| `reports/` | Measured synthetic demonstration results |
| `docs/interview-guide.md` | Plain-English explanation and limitations |

## Evaluation and limitations

MAE is the mean absolute difference between actual and predicted claim cost. Relative improvement is `(linear_MAE - xgb_MAE) / linear_MAE * 100`; a negative result means XGBoost did worse. It must be measured on the same held-out rows in original units. The code does not force any improvement percentage.

Frequency encoding can collapse different categories with equal frequency and erase useful category identity. One-hot encoding is a worthwhile future comparison. The current linear baseline deliberately uses the same preprocessing and target transformation as XGBoost; it is not proof that all linear approaches are inferior.

Log-scale squared-error training does not directly minimize original-unit MAE or estimate an unbiased arithmetic mean after inversion. Compare objectives and transformations using validation data before making business claims. The bootstrap assumes independent claims; the anonymized dataset lacks policyholder/time groups here. A deployment would need grouped/time-based evaluation, drift monitoring, calibration and a review of feature availability at prediction time.

Sources: [Scikit-Learn target transformation](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html), [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
