"""Tests for the Credit Risk API."""

import pytest
from fastapi.testclient import TestClient

import src.api.app as app_module
from src.api.app import app
from src.api.predict import CreditRiskPredictor

# Initialize predictor before creating client
app_module.predictor = CreditRiskPredictor()
client = TestClient(app)

SAMPLE_BORROWER = {
    "loan_amnt": 15000,
    "term": 36,
    "int_rate": 13.5,
    "installment": 510,
    "annual_inc": 75000,
    "dti": 18.5,
    "fico_range_low": 690,
    "fico_range_high": 694,
    "home_ownership": "MORTGAGE",
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "initial_list_status": "f",
    "application_type": "Individual",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_predict_returns_valid_response():
    response = client.post("/predict", json=SAMPLE_BORROWER)
    assert response.status_code == 200
    data = response.json()
    assert "pd" in data
    assert "lgd" in data
    assert "expected_loss" in data
    assert "risk_tier" in data
    assert "top_risk_drivers" in data


def test_predict_pd_in_range():
    response = client.post("/predict", json=SAMPLE_BORROWER)
    data = response.json()
    assert 0 <= data["pd"] <= 1


def test_predict_lgd_in_range():
    response = client.post("/predict", json=SAMPLE_BORROWER)
    data = response.json()
    assert 0 <= data["lgd"] <= 1


def test_predict_risk_tier_valid():
    response = client.post("/predict", json=SAMPLE_BORROWER)
    data = response.json()
    assert data["risk_tier"] in ["Low", "Medium", "High", "Very High"]


def test_predict_el_formula():
    response = client.post("/predict", json=SAMPLE_BORROWER)
    data = response.json()
    expected_el = data["pd"] * data["lgd"] * SAMPLE_BORROWER["loan_amnt"]
    assert abs(data["expected_loss"] - expected_el) < 1.0


def test_predict_high_risk_borrower():
    high_risk = {**SAMPLE_BORROWER, "int_rate": 28, "fico_range_low": 620,
                 "fico_range_high": 624, "term": 60, "dti": 35}
    response = client.post("/predict", json=high_risk)
    data = response.json()
    assert data["pd"] > 0.5


def test_predict_low_risk_borrower():
    low_risk = {**SAMPLE_BORROWER, "int_rate": 6, "fico_range_low": 780,
                "fico_range_high": 784, "term": 36, "dti": 5, "annual_inc": 150000}
    response = client.post("/predict", json=low_risk)
    data = response.json()
    assert data["pd"] < 0.3


def test_predict_invalid_input():
    invalid = {**SAMPLE_BORROWER, "loan_amnt": -1000}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
