from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

BQ_PROJECT = "your-gcp-project-id"
BQ_DATASET = "credit_risk"
BQ_RAW_TABLE = "raw_loans"
BQ_CLEAN_TABLE = "clean_loans"

RANDOM_SEED = 42

SELECTED_COLUMNS = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_title", "emp_length",
    "home_ownership", "annual_inc", "verification_status",
    "issue_d", "loan_status", "purpose", "title",
    "dti", "earliest_cr_line", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc",
    "initial_list_status", "application_type",
    "mort_acc", "pub_rec_bankruptcies",
    "fico_range_low", "fico_range_high",
    "last_pymnt_amnt", "total_pymnt", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_fico_range_high", "last_fico_range_low",
]

TERMINAL_STATUSES = ["Fully Paid", "Charged Off", "Default"]

DEFAULT_STATUSES = ["Charged Off", "Default"]

TRAIN_END_YEAR = 2016
TEST_START_YEAR = 2017
