# Credit Risk Analysis + LLM Reporting

End-to-end credit risk modelling pipeline using Lending Club loan data (2007–2018, 1.3M terminal loans). Combines traditional risk modelling (PD/LGD/EL), causal inference, explainability, and LLM-powered reporting.

## Key Findings

- **XGBoost PD model** achieves AUC-ROC of **0.712** (vs 0.696 for Logistic Regression) with interest rate, loan term, and DTI as the top default drivers
- **Expected Loss** framework assigns quantile-based risk tiers across a $3.3B test portfolio, with mean EL of $3,572 per loan
- **Diff-in-Diff analysis** estimates a **-4.5pp** relative reduction in 60-month loan defaults post-2016 credit tightening (p=0.058, borderline significant)
- **RAG-powered risk reports** map individual borrower assessments to Basel III IRB parameters and IFRS 9 staging

## Architecture

```
Lending Club Data (CSV)
    ↓
[Data Pipeline] → BigQuery + Parquet
    ↓
[EDA] → 18 diagnostic plots
    ↓
[PD Model] → LogReg + XGBoost (Optuna-tuned)
    ↓
[LGD Model] → XGBoost Regressor
    ↓
[Expected Loss] → EL = PD × LGD × EAD + Risk Tiers
    ↓
[SHAP] → Global + Local Explanations
    ↓
[Diff-in-Diff] → Causal Impact of Credit Tightening
    ↓
[RAG + Claude] → Basel III / IFRS 9 Risk Reports
    ↓
[FastAPI] → REST API → Docker → Cloud Run
```

## Modules

| # | Module | Key Output |
|---|--------|------------|
| 1 | Data Ingestion | 1.3M cleaned loans, BigQuery tables, 4 SQL scripts |
| 2 | EDA + Feature Engineering | 18 plots, 42 model features, time-based train/test split |
| 3 | PD Model | XGBoost AUC-ROC 0.712, LR AUC-ROC 0.696, Optuna tuning |
| 4 | LGD Model | XGBoost regressor, MAE 0.068, Expected Loss per borrower |
| 5 | SHAP Analysis | Beeswarm, waterfall, LR vs XGBoost importance comparison |
| 6 | Diff-in-Diff | -4.5pp treatment effect, parallel trends, bootstrap CI, placebo test |
| 7 | Power BI Dashboard | 4-tab data exports (portfolio, risk, vintage, policy impact) |
| 8 | RAG + Claude API | ChromaDB + sentence-transformers, Basel III/IFRS 9 risk reports |
| 9 | FastAPI + Docker | REST API, 9 passing tests, Dockerfile, CI/CD pipeline |
| 10 | README + Cleanup | This file |

## Project Structure

```
CreditRiskAnalysis/
├── config/settings.py           # Central configuration
├── src/
│   ├── data/                    # download, clean, features, BigQuery upload
│   ├── eda/                     # univariate + bivariate EDA plots
│   ├── models/                  # PD, LGD, Expected Loss, evaluation
│   ├── shap_analysis/           # global + local SHAP
│   ├── causal/                  # Diff-in-Diff analysis
│   ├── rag/                     # ChromaDB vector store + risk report generation
│   └── api/                     # FastAPI app, schemas, prediction logic
├── sql/                         # BigQuery SQL scripts (01-04)
├── models/                      # Saved .joblib models
├── outputs/                     # All plots, metrics, reports
├── dashboard/                   # Power BI data exports
├── docs/basel3/                 # Regulatory reference documents
├── tests/                       # API tests
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Local dev setup
└── .github/workflows/ci_cd.yml  # GitHub Actions CI/CD
```

## How to Run

### Setup
```bash
git clone <repo-url>
cd CreditRiskAnalysis
pip install -r requirements.txt
```

### Full Pipeline
```bash
# Module 1: Download and clean data
python -m src.data.download
python -m src.data.clean

# Module 2: Feature engineering and EDA
python -m src.data.features
python -m src.eda.univariate
python -m src.eda.bivariate

# Module 3-4: Train models
python -m src.models.pd_model
python -m src.models.lgd_model
python -m src.models.expected_loss

# Module 5: SHAP analysis
python -m src.shap_analysis.global_shap
python -m src.shap_analysis.local_shap

# Module 6: Causal inference
python -m src.causal.did_analysis

# Module 8: RAG reports
python -m src.rag.vector_store
# Then run risk report steps (see src/rag/risk_report.py)
```

### API
```bash
# Local
uvicorn src.api.app:app --port 8080

# Docker
docker compose up --build

# Test
pytest tests/ -v
```

### API Usage
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000, "term": 36, "int_rate": 13.5,
    "installment": 510, "annual_inc": 75000, "dti": 18.5,
    "fico_range_low": 690, "fico_range_high": 694
  }'
```

Response:
```json
{
  "pd": 0.4547,
  "lgd": 0.5044,
  "expected_loss": 3440.28,
  "risk_tier": "Medium",
  "top_risk_drivers": [
    {"feature": "int_rate", "impact": 0.2555},
    {"feature": "term", "impact": 0.2087}
  ]
}
```

## Tech Stack

- **ML**: scikit-learn, XGBoost, Optuna, SHAP
- **Data**: pandas, PyArrow, BigQuery
- **Causal Inference**: statsmodels (OLS + clustered SEs)
- **RAG**: ChromaDB, sentence-transformers, Anthropic Claude API
- **API**: FastAPI, Pydantic, uvicorn
- **Deployment**: Docker, Google Cloud Run, GitHub Actions
- **Visualisation**: matplotlib, seaborn, Power BI

## Data

Lending Club Loan Data (2007–2018) via Kaggle: 2.2M loans, filtered to 1.3M terminal-status loans (Fully Paid + Charged Off/Default). Time-based split: train on 2007–2016, test on 2017–2018.

Source: [Lending Club dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
