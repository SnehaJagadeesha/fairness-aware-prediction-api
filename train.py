"""Trains the model on a SYNTHETIC loan dataset and reports fairness metrics."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core import FEATURES, METRICS_PATH, MODEL_PATH


def make_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], size=n, p=[0.6, 0.4])
    is_b = (group == "B").astype(float)
    age = rng.integers(21, 65, n)
    income = np.clip(rng.normal(55000, 15000, n) - is_b * 9000, 12000, None)
    credit = np.clip(rng.normal(650, 70, n) - is_b * 20, 300, 850).round()
    years = np.clip(rng.normal(6, 4, n), 0, 40)
    score = (
        0.00005 * (income - 50000)
        + 0.010 * (credit - 650)
        + 0.04 * years
        + rng.normal(0, 0.6, n)
    )
    approved = (score > 0.1).astype(int)
    return pd.DataFrame(
        {
            "age": age, "income": income.round(2), "credit_score": credit,
            "years_employed": years.round(1), "group": group, "approved": approved,
        }
    )


def main() -> None:
    df = make_data()
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df["approved"]
    )
    # The sensitive attribute ("group") is NOT used as a model feature.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train_df[FEATURES], train_df["approved"])

    predictions = model.predict(test_df[FEATURES])
    accuracy = accuracy_score(test_df["approved"], predictions)

    # Fairness check: how often is each group approved?
    rates = (
        pd.Series(predictions, index=test_df.index).groupby(test_df["group"]).mean()
    )
    parity_gap = float(abs(rates["A"] - rates["B"]))
    impact_ratio = float(rates.min() / rates.max())

    metrics = {
        "dataset": "synthetic loan-approval data (5000 rows)",
        "accuracy": round(float(accuracy), 4),
        "approval_rate_group_A": round(float(rates["A"]), 4),
        "approval_rate_group_B": round(float(rates["B"]), 4),
        "demographic_parity_difference": round(parity_gap, 4),
        "disparate_impact_ratio": round(impact_ratio, 4),
        "note": "A disparate impact ratio below 0.8 may indicate adverse impact "
        "(the 'four-fifths rule').",
    }
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
