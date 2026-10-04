# Decision Log

Key technical decisions made during the Credit Risk Analysis project and the reasoning behind them.

## Data

### Time-based train/test split (2007–2016 / 2017–2018)
Random splits leak future information into training — a model trained on 2018 loans and tested on 2014 loans is unrealistic. Time-based splitting mirrors how the model would be deployed: trained on historical data, predicting on future applicants. This also surfaces temporal distribution shift in evaluation metrics.

### Terminal statuses only (Fully Paid + Charged Off/Default)
Current and In Grace Period loans have unknown outcomes. Including them would require survival analysis or right-censoring. Filtering to terminal statuses gives clean binary labels at the cost of dropping ~40% of data.

### 38 selected columns from 150+
Most of the 150 columns are either post-origination (total_pymnt, last_pymnt_amnt), free-text (desc, emp_title), or highly redundant. Selected columns cover the features available at loan origination that a lender would use for underwriting.

## PD Model

### XGBoost + Logistic Regression dual approach
LR provides interpretable coefficients and a regulatory-friendly baseline. XGBoost captures non-linear interactions (e.g., int_rate × term). Comparing both shows where the performance gap justifies model complexity — in this case, ~1.6pp AUC-ROC difference (0.712 vs 0.696).

### scale_pos_weight over SMOTE
SMOTE creates synthetic samples that can introduce noise, especially in high-dimensional spaces. `scale_pos_weight` adjusts the loss function directly without altering the training distribution. Simpler and faster on 1M+ rows.

### Optuna with 15 trials on 200K subsample
Full-dataset hyperparameter search (1M rows × 100+ trials) would take hours. Subsampling 200K rows for tuning, then training the final model on the full dataset, gives good hyperparameters in minutes. 15 trials with Optuna's TPE sampler is more efficient than grid search.

### solver='saga' for Logistic Regression
The default `lbfgs` solver is slow on datasets over 100K rows. `saga` supports L1/L2/ElasticNet, handles large-n efficiently, and converges reliably with `max_iter=500`.

## LGD Model

### Recovery-based target (1 - recoveries/loan_amnt)
LGD = 1 − recovery_rate is the standard Basel II/III definition. Using `recoveries / loan_amnt` instead of `total_rec_prncp / funded_amnt` because recoveries captures post-charge-off collections, which is what LGD represents.

### Excluding recovery_rate and lgd from features
These columns are derived from the target variable. Including them caused R² = 0.99 (data leakage). After exclusion, R² = -0.04 with MAE = 0.068 — realistic for LGD prediction where most of the variance is driven by unobservable workout-process factors.

## Expected Loss

### Quantile-based risk tiers over fixed thresholds
XGBoost's `scale_pos_weight` shifts the PD distribution upward (mean PD = 0.47 vs actual default rate = 0.21). Fixed thresholds [0.02, 0.05, 0.10] classified 96% of loans as "Very High". Quantile-based tiers (`pd.qcut` with 4 equal bins) produce a balanced distribution that's more useful for portfolio segmentation.

## Causal Inference (Diff-in-Diff)

### 60-month vs 36-month as treatment/control
Lending Club's 2016 credit tightening disproportionately affected longer-term loans (more exposure, higher risk). 36-month loans serve as a natural control group — same platform, same borrowers, less affected by the policy change.

### Honest reporting of parallel trends violation
The placebo test (fake treatment at 2013) was significant (p=0.0065), indicating the parallel trends assumption doesn't fully hold. Rather than dropping the analysis or cherry-picking a different specification, we report the DiD result (-4.5pp, p=0.058) alongside this limitation. The analysis still demonstrates the methodology.

## RAG Pipeline

### Two-step process (separate Python processes)
Loading XGBoost models and sentence-transformers in the same process caused SIGSEGV (exit code 139) due to conflicting native library dependencies. Splitting into Step 1 (ML inference → JSON) and Step 2 (embedding + ChromaDB + Claude API) avoids the segfault entirely.

### all-MiniLM-L6-v2 embedding model
Lightweight (80MB), fast inference, good performance on semantic similarity benchmarks. Overkill to use a larger model for ~12 document chunks.

### Template fallback for risk reports
If the Claude API key isn't set, the pipeline generates a deterministic template report with the same structure. This ensures the pipeline works end-to-end without requiring API credits for testing.

## API

### FastAPI over Flask
FastAPI provides automatic Pydantic validation, OpenAPI docs, and async support out of the box. For a model-serving endpoint, the automatic request validation and documentation are particularly valuable.

### Direct predictor initialization in tests
FastAPI's lifespan context manager doesn't run with TestClient. Instead of adding test-specific startup logic, we initialize the predictor directly: `app_module.predictor = CreditRiskPredictor()`. Simple and avoids test infrastructure coupling.

### Port 8080 for Cloud Run compatibility
Cloud Run defaults to port 8080. Using it everywhere (local, Docker, CI/CD) avoids port mapping confusion.
