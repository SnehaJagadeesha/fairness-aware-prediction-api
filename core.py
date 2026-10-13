"""Core logic: model loading, predictions and the SQLite log (no web code here)."""
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import joblib
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model.joblib")
METRICS_PATH = os.path.join(BASE, "metrics.json")
FEATURES = ["age", "income", "credit_score", "years_employed"]

_model = None


def db_path() -> str:
    return os.environ.get("DB_PATH", os.path.join(BASE, "predictions.db"))


def init_db() -> None:
    with closing(sqlite3.connect(db_path())) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                age INTEGER, income REAL, credit_score INTEGER,
                years_employed REAL, approved INTEGER, probability REAL
            )"""
        )
        conn.commit()


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Run: python train.py")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(data: dict) -> dict:
    model = load_model()
    row = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    probability = float(model.predict_proba(row)[0][1])
    approved = probability >= 0.5
    save_prediction(data, approved, probability)
    return {"approved": approved, "probability": round(probability, 4)}


def save_prediction(data: dict, approved: bool, probability: float) -> None:
    with closing(sqlite3.connect(db_path())) as conn:
        conn.execute(
            "INSERT INTO predictions (created_at, age, income, credit_score, "
            "years_employed, approved, probability) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                data["age"], data["income"], data["credit_score"],
                data["years_employed"], int(approved), probability,
            ),
        )
        conn.commit()


def recent_predictions(limit: int = 10) -> list:
    with closing(sqlite3.connect(db_path())) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def load_fairness_report() -> dict:
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError("Metrics not found. Run: python train.py")
    with open(METRICS_PATH) as f:
        return json.load(f)
