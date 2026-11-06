"""Strict CSV contract and transactional SQLite ingestion (no external server)."""
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .features import CAT, NUM, FEATURES


def validate(frame, training=True):
    required = ["id"] + FEATURES + (["loss"] if training else [])
    missing = sorted(set(required) - set(frame))
    extra = sorted(set(frame) - set(required) - {"loss"})
    if missing or extra:
        raise ValueError(f"Schema mismatch: missing={missing}, unexpected={extra}")
    frame = frame[required].copy()
    if frame.empty:
        raise ValueError("Input contains no claims")
    # Reject duplicate IDs rather than silently drop potentially conflicting claims.
    if frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("Claim IDs must be present and unique; duplicate IDs detected")
    ids = pd.to_numeric(frame.id, errors="raise")
    if not np.isfinite(ids).all() or (ids % 1 != 0).any() or (ids <= 0).any():
        raise ValueError("IDs must be positive finite integers")
    frame["id"] = ids.astype("int64")
    if frame.id.duplicated().any():
        raise ValueError("Duplicate IDs after numeric conversion")
    for col in NUM + (["loss"] if training else []):
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
        if np.isinf(frame[col]).any():
            raise ValueError(f"Infinite numeric value in {col}")
    if training and (frame.loss.isna().any() or (frame.loss < 0).any()):
        raise ValueError("Target loss must be present, finite and nonnegative")
    for col in CAT:
        nonmissing = frame[col].dropna()
        if not nonmissing.map(lambda v: isinstance(v, str)).all():
            raise ValueError(f"{col} must contain strings or missing values")
        frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)
    # Exact duplicate examples with different IDs could leak across a random split.
    if training and frame.drop(columns="id").duplicated().any():
        raise ValueError("Duplicate claim records with different IDs detected")
    report = {
        "rows": len(frame), "categorical_features": len(CAT),
        "numerical_features": len(NUM), "duplicate_ids": 0,
        "missing_by_column": {k: int(v) for k, v in frame.isna().sum().items()},
    }
    return frame, report


def read_csv(path, training=True):
    # Read category tokens verbatim: a category such as NA is not a missing value.
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={c: str for c in CAT})
    return validate(frame, training)


def ingest(csv_path, db_path, source):
    frame, report = read_csv(csv_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    schema = (Path(__file__).parents[1] / "sql" / "schema.sql").read_text()
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(schema)
        # All tables are committed together; repeated identical inputs are no-ops.
        prior = conn.execute("SELECT batch_id FROM ingestion_batches WHERE sha256 = ?", (fingerprint,)).fetchone()
        if prior:
            return prior[0], report
        with conn:
            cur = conn.execute(
                "INSERT INTO ingestion_batches(source,sha256,row_count,quality_json) VALUES (?,?,?,?)",
                (source, fingerprint, len(frame), json.dumps(report)))
            batch = cur.lastrowid
            conn.executemany("INSERT INTO claims(batch_id,claim_id,loss) VALUES (?,?,?)",
                             ((batch, int(r.id), float(r.loss)) for r in frame.itertuples()))
            columns = ",".join(["batch_id", "claim_id"] + FEATURES)
            marks = ",".join(["?"] * (len(FEATURES) + 2))
            # Feature names come exclusively from our fixed schema, never user input.
            rows = frame[["id"] + FEATURES].astype(object).where(frame[["id"] + FEATURES].notna(), None)
            conn.executemany(f"INSERT INTO claim_features({columns}) VALUES ({marks})",
                             ((batch, *r) for r in rows.itertuples(index=False, name=None)))
    return batch, report


def load_batch(db_path, batch):
    cols = ",".join(f"f.{c}" for c in FEATURES)
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            f"SELECT c.claim_id AS id,c.loss,{cols} FROM claims c JOIN claim_features f "
            "ON c.batch_id=f.batch_id AND c.claim_id=f.claim_id WHERE c.batch_id=? ORDER BY c.claim_id",
            conn, params=(batch,))


def synthetic(n=5000, seed=42):
    """Artificial data for smoke testing; NOT Allstate or a business benchmark."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.choice(["A", "B", "C"], n, p=[.65, .25, .10]) for c in CAT})
    for col in NUM:
        frame[col] = rng.uniform(0, 1, n)
    log_loss = (7 + 1.2 * (frame.cat1 == "C") + .8 * (frame.cont1 > .6)
                + 1.4 * frame.cont2 ** 2 + .7 * frame.cont3 * frame.cont4
                + rng.normal(0, .35, n))
    frame["loss"] = np.expm1(log_loss)
    frame.insert(0, "id", np.arange(1, n + 1))
    frame.loc[rng.random(n) < .01, "cont5"] = np.nan
    frame.loc[rng.random(n) < .01, "cat5"] = np.nan
    return frame
