# Fairness-Aware Prediction API

A REST API (FastAPI) that serves a loan-approval model and reports how fairly it
treats two groups. It covers model training, an API, SQL logging and automated tests.

Note: the model is trained on a synthetic dataset generated in train.py.
It is a demonstration project, not a real lending system.

## Endpoints
- POST /predict - approval decision and probability (inputs are validated)
- GET /predictions - latest predictions, stored in SQLite
- GET /fairness - approval rate per group, demographic parity difference and
  disparate impact ratio (four-fifths rule check)
- GET /health - health check

## Tech stack
Python, FastAPI, scikit-learn, pandas, SQLite, pytest, Git, GitHub

## Run it
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python train.py
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive API page.

## Run the tests
    python train.py && pytest -v

## About the fairness report
The sensitive attribute (group) is not a model feature, but the groups differ in
income and credit score in the synthetic data, so approval rates still differ.
The report measures this gap; it does not fix it. Bias mitigation is a next step.
