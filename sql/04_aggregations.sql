-- Aggregation queries for EDA and dashboarding

-- Default rate by grade
SELECT
    grade,
    COUNT(*) AS total_loans,
    COUNTIF(default_flag = 1) AS defaults,
    ROUND(COUNTIF(default_flag = 1) / COUNT(*) * 100, 2) AS default_rate_pct,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amt,
    ROUND(AVG(int_rate), 2) AS avg_int_rate
FROM `credit_risk.feature_loans`
GROUP BY grade
ORDER BY grade;

-- Default rate by FICO band
SELECT
    fico_band,
    COUNT(*) AS total_loans,
    ROUND(COUNTIF(default_flag = 1) / COUNT(*) * 100, 2) AS default_rate_pct,
    ROUND(AVG(fico_avg), 0) AS avg_fico
FROM `credit_risk.feature_loans`
GROUP BY fico_band
ORDER BY avg_fico DESC;

-- Default rate trend by issue year and term (for diff-in-diff)
SELECT
    issue_year,
    term,
    COUNT(*) AS total_loans,
    COUNTIF(default_flag = 1) AS defaults,
    ROUND(COUNTIF(default_flag = 1) / COUNT(*) * 100, 2) AS default_rate_pct
FROM `credit_risk.feature_loans`
GROUP BY issue_year, term
ORDER BY issue_year, term;

-- Monthly default rate by vintage
SELECT
    issue_year,
    issue_month,
    COUNT(*) AS total_loans,
    ROUND(COUNTIF(default_flag = 1) / COUNT(*) * 100, 2) AS default_rate_pct,
    ROUND(SUM(loan_amnt), 0) AS total_exposure
FROM `credit_risk.feature_loans`
GROUP BY issue_year, issue_month
ORDER BY issue_year, issue_month;

-- Portfolio summary
SELECT
    COUNT(*) AS total_loans,
    ROUND(SUM(loan_amnt), 0) AS total_exposure,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amt,
    ROUND(COUNTIF(default_flag = 1) / COUNT(*) * 100, 2) AS overall_default_rate_pct,
    ROUND(AVG(int_rate), 2) AS avg_int_rate,
    ROUND(AVG(dti), 2) AS avg_dti,
    ROUND(AVG(fico_avg), 0) AS avg_fico
FROM `credit_risk.feature_loans`;
