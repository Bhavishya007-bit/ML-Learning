"""Train/validation/test evaluation with an untouched test split."""
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .features import FEATURES, preprocessor


def build_model(estimator, n_train, seed):
    return TransformedTargetRegressor(
        regressor=Pipeline([("prepare", preprocessor(n_train, seed)), ("model", estimator)]),
        func=np.log1p, inverse_func=np.expm1)


def predict(model, X):
    # Nonnegative loss constraint, identical for baseline and boosted models.
    result = np.maximum(0, model.predict(X))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite predictions; inspect training data and model")
    return result


def metrics(actual, predicted):
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(root_mean_squared_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted))}


def segment(predictions, thresholds):
    low, high = thresholds
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError("Invalid severity-band thresholds")
    return np.where(predictions <= low, "low", np.where(predictions <= high, "medium", "high"))


def train(frame, output, source, seed=42):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if len(frame) < 100:
        raise ValueError("At least 100 rows required for a three-way split")
    trainval, test = train_test_split(frame, test_size=.20, random_state=seed)
    fit, valid = train_test_split(trainval, test_size=.25, random_state=seed)
    # 60/20/20. No encoders, imputation values or quantiles are learned on test.
    split_ids = pd.concat([part[["id"]].assign(split=name) for name, part in
                           [("train", fit), ("validation", valid), ("test", test)]])
    split_ids.to_csv(out / "split_ids.csv", index=False)
    linear = build_model(LinearRegression(), len(fit), seed)
    linear.fit(fit[FEATURES], fit.loss)
    dummy = DummyRegressor(strategy="median").fit(fit[FEATURES], fit.loss)
    candidates = []
    best = None
    # Small declared search, chosen by validation MAE in ORIGINAL loss units.
    for depth in [3, 5]:
        config = dict(n_estimators=250, max_depth=depth, learning_rate=.05,
                      subsample=.9, colsample_bytree=.9, reg_lambda=5,
                      objective="reg:squarederror", tree_method="hist",
                      n_jobs=2, random_state=seed)
        candidate = build_model(XGBRegressor(**config), len(fit), seed)
        candidate.fit(fit[FEATURES], fit.loss)
        score = metrics(valid.loss, predict(candidate, valid[FEATURES]))
        candidates.append({"parameters": config, "validation": score})
        if best is None or score["mae"] < best[0]:
            best = (score["mae"], candidate, config)
    xgb = best[1]
    validation = {
        "median": metrics(valid.loss, dummy.predict(valid[FEATURES])),
        "linear_regression": metrics(valid.loss, predict(linear, valid[FEATURES])),
        "xgboost": metrics(valid.loss, predict(xgb, valid[FEATURES])),
    }
    # Freeze cutoffs on validation predictions, not on test labels or test quantiles.
    thresholds = np.quantile(predict(xgb, valid[FEATURES]), [1/3, 2/3]).tolist()
    models = {"median": dummy, "linear_regression": linear, "xgboost": xgb}
    test_scores = {}
    predictions = test[["id", "loss"]].copy()
    for name, model in models.items():
        pred = predict(model, test[FEATURES])
        test_scores[name] = metrics(test.loss, pred)
        predictions[name] = pred
    predictions["severity_band"] = segment(predictions.xgboost, thresholds)
    predictions["absolute_error"] = abs(predictions.loss - predictions.xgboost)
    predictions.to_csv(out / "test_predictions.csv", index=False)
    summary = predictions.groupby("severity_band").agg(
        claims=("id", "size"), predicted_mean=("xgboost", "mean"),
        actual_mean=("loss", "mean"), actual_median=("loss", "median"),
        mae=("absolute_error", "mean"))
    summary.reindex(["low", "medium", "high"]).to_csv(out / "segment_summary.csv")
    base_mae = test_scores["linear_regression"]["mae"]
    improvement = 100 * (base_mae - test_scores["xgboost"]["mae"]) / base_mae if base_mae else None
    # Paired bootstrap quantifies uncertainty in the test MAE difference.
    base_errors = abs(predictions.loss - predictions.linear_regression).to_numpy()
    xgb_errors = predictions.absolute_error.to_numpy()
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(500):
        idx = rng.integers(0, len(test), len(test))
        differences.append(float(np.mean(base_errors[idx] - xgb_errors[idx])))
    report = {
        "source": source, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic": source == "synthetic_demo", "seed": seed,
        "rows": len(frame), "features": len(FEATURES),
        "split_counts": {"train": len(fit), "validation": len(valid), "test": len(test)},
        "validation": validation, "xgboost_candidates": candidates,
        "selected_parameters": best[2], "test": test_scores,
        "xgboost_mae_reduction_percent_vs_linear": improvement,
        "mae_difference_bootstrap_95_percent_interval": np.quantile(differences, [.025, .975]).tolist(),
        "severity_band_thresholds": thresholds,
        "selection": "XGBoost depth selected on validation MAE; test evaluated after selection. No refit.",
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]},
        "python": platform.python_version(),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    joblib.dump({"model": xgb, "thresholds": thresholds, "features": FEATURES,
                 "source": source}, out / "model.joblib")
    lines = ["# Evaluation report", "", f"Data source: **{source}**", "",
             "Synthetic data results are software demonstrations, not insurance benchmarks." if report["synthetic"]
             else "Local holdout results; not a Kaggle leaderboard score.", "",
             f"Rows: {len(frame):,}; features: {len(FEATURES)}; split: 60%/20%/20%.", "",
             "| Model | Test MAE | Test RMSE | Test R² |", "|---|---:|---:|---:|"]
    for name, score in test_scores.items():
        lines.append(f"| {name} | {score['mae']:.2f} | {score['rmse']:.2f} | {score['r2']:.3f} |")
    lines += ["", f"Measured MAE reduction versus linear: {improvement:.2f}%." if improvement is not None
              else "Relative MAE reduction undefined: baseline MAE is zero.", "",
              "Both regressions use log1p(loss); all metrics above use original loss units.",
              "See metrics.json for parameters, versions, validation scores and bootstrap uncertainty."]
    (out / "evaluation.md").write_text("\n".join(lines) + "\n")
    return report
