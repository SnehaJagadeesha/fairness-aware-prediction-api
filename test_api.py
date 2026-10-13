import os
import tempfile

# Use a throwaway database for tests (must be set before importing main).
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

client = TestClient(main.app)

GOOD = {"age": 35, "income": 90000, "credit_score": 760, "years_employed": 8}
BAD = {"age": 22, "income": 18000, "credit_score": 520, "years_employed": 0.5}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_approves_strong_applicant():
    response = client.post("/predict", json=GOOD)
    assert response.status_code == 200
    assert response.json()["approved"] is True


def test_predict_rejects_weak_applicant():
    response = client.post("/predict", json=BAD)
    assert response.status_code == 200
    assert response.json()["approved"] is False


def test_predict_rejects_invalid_input():
    response = client.post("/predict", json={**GOOD, "credit_score": 900})
    assert response.status_code == 422


def test_predictions_are_saved():
    client.post("/predict", json=GOOD)
    response = client.get("/predictions?limit=5")
    assert response.status_code == 200
    assert len(response.json()) >= 1


def test_fairness_report():
    response = client.get("/fairness")
    assert response.status_code == 200
    assert "disparate_impact_ratio" in response.json()
