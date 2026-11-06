import sqlite3
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from claims.data import ingest, load_batch, synthetic, validate
from claims.features import FEATURES, FrequencyEncoder
from claims.model import build_model, predict, segment, train


def test_encoder_uses_training_only_and_handles_unknown_missing():
    encoder = FrequencyEncoder().fit(pd.DataFrame({"cat": ["A", "A", "B", None]}))
    transformed = encoder.transform(pd.DataFrame({"cat": ["B", "NEW", None]}))
    np.testing.assert_allclose(transformed.ravel(), [.25, 0, .25])
    assert encoder.maps_[0]["A"] == .5


@pytest.mark.parametrize("bad", ["id_duplicate", "loss_missing", "negative_loss", "infinity", "bad_numeric", "missing_column", "record_duplicate"])
def test_bad_inputs_rejected(bad):
    frame = synthetic(30)
    if bad == "id_duplicate": frame.loc[1, "id"] = frame.loc[0, "id"]
    if bad == "loss_missing": frame.loc[0, "loss"] = np.nan
    if bad == "negative_loss": frame.loc[0, "loss"] = -1
    if bad == "infinity": frame.loc[0, "cont1"] = np.inf
    if bad == "bad_numeric": frame["cont1"] = "broken"
    if bad == "missing_column": frame = frame.drop(columns="cat1")
    if bad == "record_duplicate": frame.loc[1, FEATURES + ["loss"]] = frame.loc[0, FEATURES + ["loss"]]
    with pytest.raises((ValueError, TypeError)):
        validate(frame)


def test_sql_roundtrip_idempotency_foreign_keys(tmp_path):
    csv, db = tmp_path / "input.csv", tmp_path / "claims.sqlite"
    frame = synthetic(100)
    frame.to_csv(csv, index=False)
    batch, report = ingest(csv, db, "test")
    assert report["rows"] == 100
    assert ingest(csv, db, "test")[0] == batch
    loaded = load_batch(db, batch)
    np.testing.assert_allclose(loaded.loss, frame.loss)
    assert list(loaded.columns) == ["id", "loss"] + FEATURES
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        assert conn.execute("SELECT count(*) FROM claims").fetchone()[0] == 100
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO claims VALUES (999,1,2)")


def test_no_preprocessing_leakage_and_serialization(tmp_path):
    frame = synthetic(100)
    frame["cat1"] = "TRAIN"
    model = build_model(LinearRegression(), 100, 42).fit(frame[FEATURES], frame.loss)
    unseen = frame[FEATURES].copy()
    unseen["cat1"] = "TEST_ONLY"
    transformed = model.regressor_.named_steps["prepare"].transform(unseen)
    assert (transformed[:, 0] == 0).all()
    joblib.dump(model, tmp_path / "model.joblib")
    np.testing.assert_allclose(predict(model, unseen), predict(joblib.load(tmp_path / "model.joblib"), unseen))


def test_segment_boundaries():
    assert segment(np.array([0, 10, 11, 20, 21]), [10, 20]).tolist() == ["low", "low", "medium", "medium", "high"]


def test_full_training_has_disjoint_splits_and_original_unit_mae(tmp_path):
    report = train(synthetic(200), tmp_path, "synthetic_demo")
    splits = pd.read_csv(tmp_path / "split_ids.csv")
    assert splits.id.is_unique and len(splits) == 200
    assert splits.groupby("split").size().to_dict() == {"test": 40, "train": 120, "validation": 40}
    preds = pd.read_csv(tmp_path / "test_predictions.csv")
    assert report["test"]["xgboost"]["mae"] == pytest.approx(abs(preds.loss - preds.xgboost).mean())
    assert report["synthetic"] is True
    assert len(report["severity_band_thresholds"]) == 2
