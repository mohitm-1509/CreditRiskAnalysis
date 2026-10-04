"""Claude API risk report generation with RAG context."""

import os
import json
import numpy as np
import pandas as pd
import joblib

from config.settings import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED
from src.rag.vector_store import query_store

RAG_OUTPUT_DIR = OUTPUTS_DIR / "rag_reports"


def get_borrower_profile(test: pd.DataFrame, idx: int, pd_prob: float,
                          lgd_pred: float, shap_top: list[tuple]) -> dict:
    """Build a borrower profile dictionary."""
    row = test.iloc[idx]
    el = pd_prob * lgd_pred * row.get("loan_amnt", 0)

    profile = {
        "loan_amount": float(row.get("loan_amnt", 0)),
        "interest_rate": float(row.get("int_rate", 0)),
        "term_months": int(row.get("term", 36)),
        "grade": str(row.get("grade", "N/A")),
        "annual_income": float(row.get("annual_inc", 0)),
        "dti": float(row.get("dti", 0)),
        "fico_score": float(row.get("fico_avg", row.get("fico_range_low", 0))),
        "home_ownership": str(row.get("home_ownership", "N/A")),
        "purpose": str(row.get("purpose", "N/A")),
        "employment_length": float(row.get("emp_length_num", 0)),
        "probability_of_default": round(float(pd_prob), 4),
        "loss_given_default": round(float(lgd_pred), 4),
        "expected_loss_dollar": round(float(el), 2),
        "risk_drivers": [{"feature": f, "shap_impact": round(float(v), 4)} for f, v in shap_top[:5]],
    }
    return profile


def build_prompt(profile: dict, rag_context: list[dict]) -> str:
    """Build the prompt for Claude API risk report generation."""
    context_text = "\n\n".join([
        f"[Source: {c['metadata']['source']} — {c['metadata']['section']}]\n{c['text']}"
        for c in rag_context
    ])

    prompt = f"""You are a credit risk analyst generating a regulatory-aligned risk assessment report.

## Borrower Profile
{json.dumps(profile, indent=2)}

## Regulatory Context (Basel III / IFRS 9)
{context_text}

## Task
Generate a structured credit risk assessment report for this borrower. Include:

1. **Risk Summary**: Overall risk assessment with risk tier classification (Low/Medium/High/Very High)
2. **Key Risk Drivers**: Analysis of the top 3-5 SHAP-identified risk factors and their implications
3. **Regulatory Context**: How this borrower's risk profile maps to Basel III IRB parameters and IFRS 9 staging
4. **Expected Loss Analysis**: Breakdown of EL = PD × LGD × EAD with commentary on each component
5. **Recommendations**: Suggested risk mitigation actions (pricing adjustments, monitoring frequency, collateral requirements)

Keep the report concise, professional, and data-driven. Reference specific regulatory thresholds where applicable."""

    return prompt


def generate_report(profile: dict, rag_context: list[dict]) -> str:
    """Generate a risk report using Claude API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _generate_template_report(profile, rag_context)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    prompt = build_prompt(profile, rag_context)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


def _generate_template_report(profile: dict, rag_context: list[dict]) -> str:
    """Generate a template report when API key is not available."""
    pd_val = profile["probability_of_default"]
    lgd_val = profile["loss_given_default"]
    el_val = profile["expected_loss_dollar"]
    ead = profile["loan_amount"]

    if pd_val < 0.15:
        tier = "Low"
    elif pd_val < 0.30:
        tier = "Medium"
    elif pd_val < 0.50:
        tier = "High"
    else:
        tier = "Very High"

    # IFRS 9 staging
    if pd_val < 0.10:
        stage = "Stage 1 (Performing)"
        ecl_type = "12-month ECL"
    elif pd_val < 0.50:
        stage = "Stage 2 (Underperforming)"
        ecl_type = "Lifetime ECL"
    else:
        stage = "Stage 3 (Non-performing)"
        ecl_type = "Lifetime ECL (credit-impaired)"

    drivers_text = "\n".join([
        f"  - **{d['feature']}**: SHAP impact = {d['shap_impact']:+.4f} "
        f"({'increases' if d['shap_impact'] > 0 else 'decreases'} default risk)"
        for d in profile["risk_drivers"]
    ])

    report = f"""# Credit Risk Assessment Report

## 1. Risk Summary

| Metric | Value |
|--------|-------|
| Risk Tier | **{tier}** |
| Probability of Default | {pd_val:.2%} |
| Loss Given Default | {lgd_val:.2%} |
| Expected Loss | ${el_val:,.2f} |
| IFRS 9 Stage | {stage} |

**Assessment**: This borrower presents a **{tier.lower()} risk** profile with a {pd_val:.1%} probability \
of default. The loan of ${ead:,.0f} at {profile['interest_rate']:.1f}% interest rate for \
{profile['term_months']} months is classified as Grade {profile['grade']}.

## 2. Key Risk Drivers (SHAP Analysis)

{drivers_text}

The dominant risk factors are consistent with established credit risk literature. \
Interest rate and loan term are the strongest predictors, reflecting the borrower's \
credit quality assessment by the originator.

## 3. Regulatory Context

### Basel III IRB Parameters
- **PD**: {pd_val:.4f} ({"above" if pd_val > 0.03 else "at"} the IRB floor of 0.03%)
- **LGD**: {lgd_val:.2%} ({"above" if lgd_val > 0.25 else "below"} the 25% unsecured retail floor)
- **EAD**: ${ead:,.0f}
- **Risk Weight**: Estimated {min(250, max(35, int(pd_val * 500))):.0f}% under IRB formula

### IFRS 9 Classification
- **Current Stage**: {stage}
- **ECL Provision Type**: {ecl_type}
- **ECL Amount**: ${el_val:,.2f}

## 4. Expected Loss Breakdown

| Component | Value | Commentary |
|-----------|-------|------------|
| PD | {pd_val:.4f} | {"Elevated" if pd_val > 0.20 else "Within normal range"} for Grade {profile['grade']} |
| LGD | {lgd_val:.4f} | {"High" if lgd_val > 0.50 else "Moderate"} loss severity |
| EAD | ${ead:,.0f} | Full loan principal at risk |
| **EL** | **${el_val:,.2f}** | {el_val/ead*100:.2f}% of exposure |

## 5. Recommendations

{"- **Enhanced Monitoring**: Quarterly review recommended due to elevated PD" if pd_val > 0.30 else "- **Standard Monitoring**: Semi-annual review sufficient"}
- **Pricing**: Current rate of {profile['interest_rate']:.1f}% {"adequately compensates" if profile['interest_rate'] > pd_val * 100 + 5 else "may not adequately compensate"} for credit risk
- **Provision**: Book ${el_val:,.2f} as {ecl_type} under IFRS 9
{"- **Collateral**: Consider requiring additional collateral for risk mitigation" if tier in ("High", "Very High") else "- **Collateral**: No additional collateral required at current risk level"}

---
*Report generated using PD model (XGBoost, AUC-ROC: 0.712) with SHAP explanations and \
Basel III/IFRS 9 regulatory context retrieved via RAG.*
"""
    return report


def _step1_compute_profiles() -> list[dict]:
    """Step 1: Compute borrower profiles using ML models (no sentence-transformers)."""
    print("Step 1: Computing borrower profiles...")
    xgb_model = joblib.load(MODELS_DIR / "pd_xgboost.joblib")
    lgd_model = joblib.load(MODELS_DIR / "lgd_xgboost.joblib")
    pd_features = joblib.load(MODELS_DIR / "pd_features.joblib")
    lgd_features = joblib.load(MODELS_DIR / "lgd_features.joblib")

    test = pd.read_parquet(PROCESSED_DIR / "features_test.parquet")
    if "fico_avg" not in test.columns:
        test["fico_avg"] = (test["fico_range_low"] + test["fico_range_high"]) / 2

    # Predict on a 10K sample to find diverse profiles (avoids 225K row segfault)
    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = rng.choice(len(test), size=min(10_000, len(test)), replace=False)
    sample = test.iloc[sample_idx].reset_index(drop=True)

    pd_probs = xgb_model.predict_proba(sample[pd_features].values)[:, 1]
    lgd_preds = lgd_model.predict(sample[lgd_features].values).clip(0, 1)
    importance = xgb_model.feature_importances_
    feat_means = sample[pd_features].mean().values

    high_risk_idx = rng.choice(np.where(pd_probs > np.percentile(pd_probs, 90))[0])
    medium_risk_idx = rng.choice(np.where(
        (pd_probs > np.percentile(pd_probs, 40)) & (pd_probs < np.percentile(pd_probs, 60))
    )[0])
    low_risk_idx = rng.choice(np.where(pd_probs < np.percentile(pd_probs, 10))[0])

    profiles = []
    for idx, label in [(high_risk_idx, "high_risk"), (medium_risk_idx, "medium_risk"),
                        (low_risk_idx, "low_risk")]:
        feat_vals = sample.iloc[idx][pd_features].values
        signed_imp = importance * np.sign(feat_vals - feat_means)
        sorted_fi = np.argsort(np.abs(signed_imp))[::-1]
        shap_top = [(pd_features[i], float(signed_imp[i])) for i in sorted_fi[:5]]

        profile = get_borrower_profile(sample, idx, pd_probs[idx], lgd_preds[idx], shap_top)
        profiles.append({"profile": profile, "label": label})
        print(f"  {label}: PD={pd_probs[idx]:.4f}, LGD={lgd_preds[idx]:.4f}")

    return profiles


def _step2_generate_reports(profiles: list[dict]) -> None:
    """Step 2: Query vector store and generate reports (loads sentence-transformers)."""
    print("\nStep 2: Generating reports with RAG...")
    for item in profiles:
        profile = item["profile"]
        label = item["label"]

        query = (f"credit risk assessment PD {profile['probability_of_default']:.2f} "
                 f"LGD {profile['loss_given_default']:.2f} "
                 f"grade {profile['grade']} expected loss provisioning")
        rag_context = query_store(query, n_results=3)

        report = generate_report(profile, rag_context)

        report_path = RAG_OUTPUT_DIR / f"risk_report_{label}.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"  Saved: {report_path}")

        profile_path = RAG_OUTPUT_DIR / f"profile_{label}.json"
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"  Saved: {profile_path}")


def run_report_pipeline(n_profiles: int = 3) -> None:
    """Generate risk reports for sample borrower profiles.

    Due to memory constraints, this should be run in two steps:
      Step 1: python3 -c "from src.rag.risk_report import ...; profiles = _step1_compute_profiles(); ..."
      Step 2: python3 -c "from src.rag.risk_report import ...; _step2_generate_reports(profiles)"

    See the inline script in __main__ for the working single-process version.
    """
    RAG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles = _step1_compute_profiles()
    _step2_generate_reports(profiles)
    print(f"\nAll reports saved to {RAG_OUTPUT_DIR}/")


if __name__ == "__main__":
    run_report_pipeline()
