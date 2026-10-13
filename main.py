"""Fairness-Aware Prediction API (FastAPI)."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import core

app = FastAPI(
    title="Fairness-Aware Prediction API",
    description="Loan-approval predictions with a built-in fairness report.",
    version="1.0.0",
)
core.init_db()


class Applicant(BaseModel):
    age: int = Field(ge=18, le=100, examples=[35])
    income: float = Field(gt=0, examples=[60000])
    credit_score: int = Field(ge=300, le=850, examples=[700])
    years_employed: float = Field(ge=0, le=60, examples=[5])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(applicant: Applicant) -> dict:
    try:
        return core.predict(applicant.model_dump())
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))


@app.get("/predictions")
def predictions(limit: int = 10) -> list:
    return core.recent_predictions(max(1, min(limit, 100)))


@app.get("/fairness")
def fairness() -> dict:
    try:
        return core.load_fairness_report()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
