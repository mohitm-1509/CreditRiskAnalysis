-- Create the BigQuery dataset and raw table schema
-- Run this after uploading via load_bq.py (which auto-creates the table)

CREATE SCHEMA IF NOT EXISTS `credit_risk`
OPTIONS (location = 'US');

-- Verify raw data loaded
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT loan_status) AS distinct_statuses,
    MIN(issue_d) AS earliest_issue,
    MAX(issue_d) AS latest_issue
FROM `credit_risk.raw_loans`;
