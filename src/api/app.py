"""FastAPI application for Credit Risk Assessment API."""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.api.schemas import BorrowerInput, PredictionResponse
from src.api.predict import CreditRiskPredictor

from typing import Optional

predictor: Optional[CreditRiskPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("Loading credit risk models...")
    predictor = CreditRiskPredictor()
    print("Models loaded successfully.")
    yield
    predictor = None


app = FastAPI(
    title="Credit Risk Assessment API",
    description="Predict probability of default, loss given default, and expected loss for consumer loans.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(borrower: BorrowerInput):
    """Predict credit risk for a borrower."""
    return predictor.predict(borrower)
