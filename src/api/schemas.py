"""Pydantic request/response schemas for the Credit Risk API."""

from pydantic import BaseModel, Field


class BorrowerInput(BaseModel):
    loan_amnt: float = Field(..., gt=0, le=40000, description="Loan amount ($)")
    term: int = Field(..., description="Loan term in months (36 or 60)")
    int_rate: float = Field(..., gt=0, le=35, description="Interest rate (%)")
    installment: float = Field(..., gt=0, description="Monthly installment ($)")
    annual_inc: float = Field(..., gt=0, description="Annual income ($)")
    dti: float = Field(..., ge=0, le=100, description="Debt-to-income ratio")
    fico_range_low: int = Field(..., ge=300, le=850, description="FICO score lower bound")
    fico_range_high: int = Field(..., ge=300, le=850, description="FICO score upper bound")
    open_acc: int = Field(default=10, ge=0, description="Number of open accounts")
    pub_rec: int = Field(default=0, ge=0, description="Public records")
    revol_bal: float = Field(default=10000, ge=0, description="Revolving balance ($)")
    revol_util: float = Field(default=50, ge=0, le=150, description="Revolving utilisation (%)")
    total_acc: int = Field(default=25, ge=0, description="Total credit accounts")
    mort_acc: int = Field(default=0, ge=0, description="Mortgage accounts")
    pub_rec_bankruptcies: float = Field(default=0, ge=0, description="Public record bankruptcies")
    emp_length_num: float = Field(default=5, ge=0, description="Employment length (years)")
    home_ownership: str = Field(default="RENT", description="Home ownership status")
    verification_status: str = Field(default="Not Verified", description="Income verification status")
    purpose: str = Field(default="debt_consolidation", description="Loan purpose")
    initial_list_status: str = Field(default="f", description="Initial listing status (f or w)")
    application_type: str = Field(default="Individual", description="Individual or Joint App")

    model_config = {"json_schema_extra": {
        "examples": [{
            "loan_amnt": 15000, "term": 36, "int_rate": 13.5, "installment": 510,
            "annual_inc": 75000, "dti": 18.5, "fico_range_low": 690, "fico_range_high": 694,
            "open_acc": 12, "pub_rec": 0, "revol_bal": 8500, "revol_util": 45,
            "total_acc": 30, "mort_acc": 1, "pub_rec_bankruptcies": 0, "emp_length_num": 7,
            "home_ownership": "MORTGAGE", "verification_status": "Verified",
            "purpose": "debt_consolidation", "initial_list_status": "f",
            "application_type": "Individual",
        }]
    }}


class RiskDriver(BaseModel):
    feature: str
    impact: float


class PredictionResponse(BaseModel):
    pd: float = Field(..., description="Probability of Default")
    lgd: float = Field(..., description="Loss Given Default")
    expected_loss: float = Field(..., description="Expected Loss ($)")
    risk_tier: str = Field(..., description="Risk tier: Low/Medium/High/Very High")
    top_risk_drivers: list[RiskDriver] = Field(..., description="Top SHAP-based risk drivers")
