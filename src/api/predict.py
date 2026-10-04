"""Prediction logic — transforms input, runs models, returns risk assessment."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.api.schemas import BorrowerInput, PredictionResponse, RiskDriver

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class CreditRiskPredictor:
    def __init__(self):
        self.pd_model = joblib.load(MODELS_DIR / "pd_xgboost.joblib")
        self.lgd_model = joblib.load(MODELS_DIR / "lgd_xgboost.joblib")
        self.pd_features = joblib.load(MODELS_DIR / "pd_features.joblib")
        self.lgd_features = joblib.load(MODELS_DIR / "lgd_features.joblib")
        self.feature_importance = self.pd_model.feature_importances_

    def _prepare_features(self, borrower: BorrowerInput) -> pd.DataFrame:
        """Transform raw borrower input into model features."""
        data = {
            "loan_amnt": borrower.loan_amnt,
            "funded_amnt": borrower.loan_amnt,
            "term": float(borrower.term),
            "int_rate": borrower.int_rate,
            "installment": borrower.installment,
            "annual_inc": borrower.annual_inc,
            "dti": max(borrower.dti, 0),
            "fico_range_low": borrower.fico_range_low,
            "fico_range_high": borrower.fico_range_high,
            "open_acc": borrower.open_acc,
            "pub_rec": borrower.pub_rec,
            "revol_bal": borrower.revol_bal,
            "revol_util": borrower.revol_util,
            "total_acc": borrower.total_acc,
            "mort_acc": borrower.mort_acc,
            "pub_rec_bankruptcies": borrower.pub_rec_bankruptcies,
            "emp_length_num": borrower.emp_length_num,
            "fico_avg": (borrower.fico_range_low + borrower.fico_range_high) / 2,
            "loan_to_income_ratio": borrower.loan_amnt / max(borrower.annual_inc, 1),
            "credit_history_years": 10.0,
        }

        cat_dummies = {
            "home_ownership": ["MORTGAGE", "NONE", "OTHER", "OWN", "RENT"],
            "verification_status": ["Source Verified", "Verified"],
            "purpose": [
                "credit_card", "debt_consolidation", "educational", "home_improvement",
                "house", "major_purchase", "medical", "moving", "other",
                "renewable_energy", "small_business", "vacation", "wedding",
            ],
            "initial_list_status": ["w"],
            "application_type": ["Joint App"],
        }

        for col, categories in cat_dummies.items():
            value = getattr(borrower, col, "")
            for cat in categories:
                feature_name = f"{col}_{cat}"
                data[feature_name] = 1 if value == cat else 0

        df = pd.DataFrame([data])

        for feat in self.pd_features:
            if feat not in df.columns:
                df[feat] = 0

        return df

    def predict(self, borrower: BorrowerInput) -> PredictionResponse:
        """Run prediction pipeline for a single borrower."""
        df = self._prepare_features(borrower)

        X_pd = df[self.pd_features].values
        X_lgd = df[self.lgd_features].values

        pd_prob = float(self.pd_model.predict_proba(X_pd)[:, 1][0])
        lgd_pred = float(np.clip(self.lgd_model.predict(X_lgd)[0], 0, 1))
        ead = borrower.loan_amnt
        el = pd_prob * lgd_pred * ead

        el_pct = pd_prob * lgd_pred
        if el_pct < 0.10:
            tier = "Low"
        elif el_pct < 0.25:
            tier = "Medium"
        elif el_pct < 0.45:
            tier = "High"
        else:
            tier = "Very High"

        feat_vals = df[self.pd_features].values[0]
        signed_imp = self.feature_importance * np.sign(feat_vals)
        sorted_idx = np.argsort(np.abs(signed_imp))[::-1][:5]
        drivers = [
            RiskDriver(feature=self.pd_features[i], impact=round(float(signed_imp[i]), 4))
            for i in sorted_idx
        ]

        return PredictionResponse(
            pd=round(pd_prob, 4),
            lgd=round(lgd_pred, 4),
            expected_loss=round(el, 2),
            risk_tier=tier,
            top_risk_drivers=drivers,
        )
