# Lending Club Column Dictionary

## Selected Columns (~35 from 150+)

### Loan Details
| Column | Description |
|--------|-------------|
| loan_amnt | The listed amount of the loan applied for |
| funded_amnt | The total amount committed to that loan |
| term | Number of payments (36 or 60 months) |
| int_rate | Interest rate on the loan (%) |
| installment | Monthly payment owed by the borrower |
| grade | LC assigned loan grade (A-G) |
| sub_grade | LC assigned loan subgrade (A1-G5) |
| purpose | Borrower's stated purpose for the loan |
| initial_list_status | Initial listing status (w = whole, f = fractional) |
| application_type | Individual or joint application |

### Borrower Profile
| Column | Description |
|--------|-------------|
| emp_title | Job title of the borrower |
| emp_length | Employment length (0-10+ years) |
| home_ownership | RENT, OWN, MORTGAGE, OTHER |
| annual_inc | Self-reported annual income |
| verification_status | Whether income was verified |
| dti | Debt-to-income ratio |

### Credit History
| Column | Description |
|--------|-------------|
| earliest_cr_line | Date of borrower's earliest credit line |
| open_acc | Number of open credit lines |
| pub_rec | Number of derogatory public records |
| revol_bal | Total revolving credit balance |
| revol_util | Revolving line utilisation rate (%) |
| total_acc | Total number of credit lines |
| mort_acc | Number of mortgage accounts |
| pub_rec_bankruptcies | Number of public record bankruptcies |
| fico_range_low | Lower boundary of FICO range at origination |
| fico_range_high | Upper boundary of FICO range at origination |

### Outcome
| Column | Description |
|--------|-------------|
| loan_status | Current status: Fully Paid, Charged Off, Default |
| total_pymnt | Payments received to date |
| total_rec_prncp | Principal received to date |
| total_rec_int | Interest received to date |
| total_rec_late_fee | Late fees received to date |
| recoveries | Post charge-off gross recovery |
| collection_recovery_fee | Post charge-off collection fee |
| last_pymnt_amnt | Last total payment amount received |
| last_fico_range_high | Upper boundary of last FICO range |
| last_fico_range_low | Lower boundary of last FICO range |

### Dates
| Column | Description |
|--------|-------------|
| issue_d | Month the loan was funded |

## Derived Features (created in feature engineering)
| Column | Description |
|--------|-------------|
| default_flag | 1 if Charged Off/Default, 0 if Fully Paid |
| issue_year | Year extracted from issue_d |
| issue_month | Month extracted from issue_d |
| fico_avg | Average of fico_range_low and fico_range_high |
| loan_to_income_ratio | loan_amnt / annual_inc |
| credit_history_years | Years between earliest_cr_line and issue_d |
| emp_length_num | Numeric version of emp_length |
| fico_band | Binned FICO score category |
| dti_bin | Binned DTI category |
| credit_util_bin | Binned revolving utilisation category |
| income_bracket | Binned annual income category |
