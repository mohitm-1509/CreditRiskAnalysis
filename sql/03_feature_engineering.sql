-- Feature engineering in BigQuery
-- Creates derived features and bins for analysis

CREATE OR REPLACE TABLE `credit_risk.feature_loans` AS
SELECT
    *,
    -- FICO bands
    CASE
        WHEN fico_avg >= 750 THEN 'Excellent (750+)'
        WHEN fico_avg >= 700 THEN 'Good (700-749)'
        WHEN fico_avg >= 650 THEN 'Fair (650-699)'
        WHEN fico_avg >= 600 THEN 'Poor (600-649)'
        ELSE 'Very Poor (<600)'
    END AS fico_band,

    -- DTI bins
    CASE
        WHEN dti <= 10 THEN 'Low (0-10)'
        WHEN dti <= 20 THEN 'Medium (10-20)'
        WHEN dti <= 30 THEN 'High (20-30)'
        ELSE 'Very High (30+)'
    END AS dti_bin,

    -- Credit utilisation bins
    CASE
        WHEN revol_util <= 20 THEN 'Very Low (0-20%)'
        WHEN revol_util <= 40 THEN 'Low (20-40%)'
        WHEN revol_util <= 60 THEN 'Medium (40-60%)'
        WHEN revol_util <= 80 THEN 'High (60-80%)'
        ELSE 'Very High (80%+)'
    END AS credit_util_bin,

    -- Income brackets
    CASE
        WHEN annual_inc <= 30000 THEN 'Low (<30k)'
        WHEN annual_inc <= 60000 THEN 'Medium (30-60k)'
        WHEN annual_inc <= 100000 THEN 'High (60-100k)'
        ELSE 'Very High (100k+)'
    END AS income_bracket,

    -- Loan amount bins
    CASE
        WHEN loan_amnt <= 5000 THEN 'Small (0-5k)'
        WHEN loan_amnt <= 15000 THEN 'Medium (5-15k)'
        WHEN loan_amnt <= 25000 THEN 'Large (15-25k)'
        ELSE 'Very Large (25k+)'
    END AS loan_amt_bin

FROM `credit_risk.clean_loans`;
