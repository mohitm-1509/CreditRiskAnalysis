-- Clean and filter the raw loans data
-- Keeps only terminal loan statuses, parses fields, handles nulls

CREATE OR REPLACE TABLE `credit_risk.clean_loans` AS
WITH parsed AS (
    SELECT
        loan_amnt,
        funded_amnt,
        CAST(REGEXP_EXTRACT(CAST(term AS STRING), r'(\d+)') AS INT64) AS term,
        CAST(REPLACE(CAST(int_rate AS STRING), '%', '') AS FLOAT64) AS int_rate,
        installment,
        grade,
        sub_grade,
        emp_length,
        CASE
            WHEN emp_length = '< 1 year' THEN 0
            WHEN emp_length = '1 year' THEN 1
            WHEN emp_length = '10+ years' THEN 10
            WHEN emp_length IS NULL THEN NULL
            ELSE CAST(REGEXP_EXTRACT(emp_length, r'(\d+)') AS INT64)
        END AS emp_length_num,
        home_ownership,
        annual_inc,
        verification_status,
        PARSE_DATE('%b-%Y', issue_d) AS issue_d,
        EXTRACT(YEAR FROM PARSE_DATE('%b-%Y', issue_d)) AS issue_year,
        EXTRACT(MONTH FROM PARSE_DATE('%b-%Y', issue_d)) AS issue_month,
        loan_status,
        CASE
            WHEN loan_status IN ('Charged Off', 'Default') THEN 1
            ELSE 0
        END AS default_flag,
        purpose,
        dti,
        PARSE_DATE('%b-%Y', earliest_cr_line) AS earliest_cr_line,
        open_acc,
        pub_rec,
        revol_bal,
        CAST(REPLACE(CAST(revol_util AS STRING), '%', '') AS FLOAT64) AS revol_util,
        total_acc,
        initial_list_status,
        application_type,
        mort_acc,
        pub_rec_bankruptcies,
        fico_range_low,
        fico_range_high,
        (fico_range_low + fico_range_high) / 2.0 AS fico_avg,
        last_pymnt_amnt,
        total_pymnt,
        total_rec_prncp,
        total_rec_int,
        total_rec_late_fee,
        recoveries,
        collection_recovery_fee,
        last_fico_range_high,
        last_fico_range_low
    FROM `credit_risk.raw_loans`
    WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
)
SELECT
    *,
    DATE_DIFF(issue_d, earliest_cr_line, MONTH) / 12.0 AS credit_history_years,
    loan_amnt / NULLIF(annual_inc, 0) AS loan_to_income_ratio
FROM parsed;

-- Verify
SELECT
    COUNT(*) AS total_rows,
    COUNTIF(default_flag = 1) AS defaults,
    COUNTIF(default_flag = 0) AS fully_paid,
    ROUND(COUNTIF(default_flag = 1) / COUNT(*) * 100, 2) AS default_rate_pct
FROM `credit_risk.clean_loans`;
