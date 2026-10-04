"""Probability of Default models — Logistic Regression + XGBoost."""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna

from config.settings import PROCESSED_DIR, MODELS_DIR, RANDOM_SEED
from src.data.features import get_model_features
from src.models.evaluation import (
    setup_style,
    compute_metrics,
    find_optimal_threshold,
    compute_ks_statistic,
    plot_roc_curves,
    plot_pr_curves,
    plot_calibration_curves,
    plot_confusion_matrix,
    plot_score_distribution,
    plot_lift_chart,
    print_comparison_table,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_SAMPLE_SIZE = 200_000


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load feature-engineered train and test sets."""
    train = pd.read_parquet(PROCESSED_DIR / "features_train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "features_test.parquet")
    features = get_model_features(train)
    print(f"Train: {len(train):,} rows, Test: {len(test):,} rows")
    print(f"Features: {len(features)}")
    print(f"Train default rate: {train['default_flag'].mean():.2%}")
    print(f"Test default rate:  {test['default_flag'].mean():.2%}")
    return train, test, features


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]
) -> Pipeline:
    """Train a Logistic Regression with StandardScaler."""
    print("\n--- Training Logistic Regression ---")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=300,
            solver="saga",
            C=0.1,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tol=1e-3,
        )),
    ])
    pipeline.fit(X_train, y_train)

    coefs = pipeline.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)

    print("\nTop 15 features by |coefficient|:")
    print(coef_df.head(15).to_string(index=False))

    return pipeline


def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                 scale_pos_weight: float, n_trials: int = 15) -> dict:
    """Tune XGBoost hyperparameters with Optuna on a subsample."""
    n = len(X_train)
    if n > TUNING_SAMPLE_SIZE:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(n, size=TUNING_SAMPLE_SIZE, replace=False)
        idx.sort()
        X_tune = X_train[idx]
        y_tune = y_train[idx]
        print(f"\n--- Tuning XGBoost ({n_trials} trials on {TUNING_SAMPLE_SIZE:,} subsample) ---")
    else:
        X_tune = X_train
        y_tune = y_train
        print(f"\n--- Tuning XGBoost ({n_trials} trials) ---")

    X_t, X_v, y_t, y_v = train_test_split(
        X_tune, y_tune, test_size=0.2, random_state=RANDOM_SEED, stratify=y_tune,
    )

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "n_estimators": 300,
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "aucpr",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "early_stopping_rounds": 30,
            "tree_method": "hist",
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)

        y_prob = model.predict_proba(X_v)[:, 1]
        return average_precision_score(y_v, y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"Best AUC-PR (val): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  best_params: dict, scale_pos_weight: float) -> xgb.XGBClassifier:
    """Train XGBoost with tuned parameters on full training data."""
    print("\n--- Training XGBoost (final model on full data) ---")

    params = {**best_params}
    params["n_estimators"] = 300
    params["scale_pos_weight"] = scale_pos_weight
    params["eval_metric"] = "aucpr"
    params["random_state"] = RANDOM_SEED
    params["n_jobs"] = -1
    params["tree_method"] = "hist"

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    print("  Training complete.")

    return model


def evaluate_models(
    models: dict, X_test: np.ndarray, y_test: np.ndarray, feature_names: list[str]
) -> dict:
    """Evaluate all models and generate comparison plots."""
    setup_style()
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    results = {}
    all_metrics = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = y_prob

        threshold = find_optimal_threshold(y_test, y_prob)
        metrics = compute_metrics(y_test, y_prob, threshold)
        ks = compute_ks_statistic(y_test, y_prob)
        metrics["ks_statistic"] = ks

        all_metrics[name] = metrics

        print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:       {metrics['auc_pr']:.4f}")
        print(f"  Gini:         {metrics['gini']:.4f}")
        print(f"  KS Statistic: {ks:.4f}")
        print(f"  Brier Score:  {metrics['brier_score']:.4f}")
        print(f"  Threshold:    {threshold:.4f}")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  F1:           {metrics['f1']:.4f}")

        plot_confusion_matrix(y_test, y_prob, threshold, name)
        plot_score_distribution(y_test, y_prob, name)
        plot_lift_chart(y_test, y_prob, name)

    plot_roc_curves(results, y_test)
    plot_pr_curves(results, y_test)
    plot_calibration_curves(results, y_test)
    print_comparison_table(all_metrics)

    return all_metrics


def run_pd_pipeline(n_tuning_trials: int = 15) -> dict:
    """Full PD modelling pipeline."""
    train, test, features = load_train_test()

    X_train = train[features].values
    y_train = train["default_flag"].values
    X_test = test[features].values
    y_test = test["default_flag"].values

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"\nClass balance: {neg_count:,} neg / {pos_count:,} pos (ratio: {scale_pos_weight:.2f})")

    lr_pipeline = train_logistic_regression(X_train, y_train, features)

    best_params = tune_xgboost(X_train, y_train, scale_pos_weight, n_trials=n_tuning_trials)

    xgb_model = train_xgboost(X_train, y_train, best_params, scale_pos_weight)

    models = {
        "Logistic Regression": lr_pipeline,
        "XGBoost": xgb_model,
    }

    all_metrics = evaluate_models(models, X_test, y_test, features)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_pipeline, MODELS_DIR / "pd_logreg.joblib")
    joblib.dump(xgb_model, MODELS_DIR / "pd_xgboost.joblib")
    joblib.dump(features, MODELS_DIR / "pd_features.joblib")
    print(f"\nModels saved to {MODELS_DIR}/")

    return all_metrics


if __name__ == "__main__":
    run_pd_pipeline(n_tuning_trials=15)
