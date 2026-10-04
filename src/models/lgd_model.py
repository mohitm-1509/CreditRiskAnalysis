"""Loss Given Default model — XGBoost regression on defaulted loans."""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED
from src.data.features import get_model_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

LGD_OUTPUT_DIR = OUTPUTS_DIR / "lgd"
TUNING_SAMPLE_SIZE = 100_000


def load_defaulted_loans() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load feature-engineered data and filter to defaulted loans only."""
    train = pd.read_parquet(PROCESSED_DIR / "features_train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "features_test.parquet")

    train_def = train[train["default_flag"] == 1].copy()
    test_def = test[test["default_flag"] == 1].copy()

    train_def["recovery_rate"] = (train_def["recoveries"] / train_def["loan_amnt"]).clip(0, 1)
    test_def["recovery_rate"] = (test_def["recoveries"] / test_def["loan_amnt"]).clip(0, 1)

    train_def["lgd"] = 1 - train_def["recovery_rate"]
    test_def["lgd"] = 1 - test_def["recovery_rate"]

    features = [f for f in get_model_features(train_def)
                if f not in ("recovery_rate", "lgd")]

    print(f"Defaulted loans — Train: {len(train_def):,}, Test: {len(test_def):,}")
    print(f"Features: {len(features)}")
    print(f"Train recovery rate: mean={train_def['recovery_rate'].mean():.4f}, "
          f"median={train_def['recovery_rate'].median():.4f}")
    print(f"Test recovery rate:  mean={test_def['recovery_rate'].mean():.4f}, "
          f"median={test_def['recovery_rate'].median():.4f}")
    print(f"Train LGD: mean={train_def['lgd'].mean():.4f}")
    print(f"Test LGD:  mean={test_def['lgd'].mean():.4f}")

    return train_def, test_def, features


def tune_lgd_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                     n_trials: int = 15) -> dict:
    """Tune XGBoost regressor hyperparameters with Optuna."""
    n = len(X_train)
    if n > TUNING_SAMPLE_SIZE:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(n, size=TUNING_SAMPLE_SIZE, replace=False)
        idx.sort()
        X_tune, y_tune = X_train[idx], y_train[idx]
        print(f"\n--- Tuning LGD XGBoost ({n_trials} trials on {TUNING_SAMPLE_SIZE:,} subsample) ---")
    else:
        X_tune, y_tune = X_train, y_train
        print(f"\n--- Tuning LGD XGBoost ({n_trials} trials) ---")

    X_t, X_v, y_t, y_v = train_test_split(
        X_tune, y_tune, test_size=0.2, random_state=RANDOM_SEED,
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
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "early_stopping_rounds": 30,
            "tree_method": "hist",
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        y_pred = model.predict(X_v).clip(0, 1)
        return mean_absolute_error(y_v, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"Best MAE (val): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def train_lgd_model(X_train: np.ndarray, y_train: np.ndarray,
                    best_params: dict) -> xgb.XGBRegressor:
    """Train final LGD XGBoost on full training data."""
    print("\n--- Training LGD XGBoost (final model on full data) ---")

    params = {**best_params}
    params["n_estimators"] = 300
    params["random_state"] = RANDOM_SEED
    params["n_jobs"] = -1
    params["tree_method"] = "hist"

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    print("  Training complete.")
    return model


def evaluate_lgd_model(model: xgb.XGBRegressor, X_test: np.ndarray,
                       y_test: np.ndarray, feature_names: list[str]) -> dict:
    """Evaluate the LGD model and generate plots."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 150, "axes.titleweight": "bold", "savefig.bbox": "tight"})
    LGD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test).clip(0, 1)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("LGD MODEL EVALUATION")
    print("=" * 50)
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Mean actual LGD:    {y_test.mean():.4f}")
    print(f"  Mean predicted LGD: {y_pred.mean():.4f}")

    # 1. Predicted vs Actual scatter
    fig, ax = plt.subplots(figsize=(8, 7))
    sample_idx = np.random.RandomState(RANDOM_SEED).choice(len(y_test), size=min(10_000, len(y_test)), replace=False)
    ax.scatter(y_test[sample_idx], y_pred[sample_idx], alpha=0.15, s=8, color="#3b82f6", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Actual LGD")
    ax.set_ylabel("Predicted LGD")
    ax.set_title(f"Predicted vs Actual LGD (R²={r2:.4f})")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(LGD_OUTPUT_DIR / "predicted_vs_actual_lgd.png")
    plt.close(fig)
    print(f"  Saved: {LGD_OUTPUT_DIR / 'predicted_vs_actual_lgd.png'}")

    # 2. LGD distribution — actual vs predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(y_test, bins=50, alpha=0.7, color="#3b82f6", edgecolor="white", density=True)
    axes[0].set_title("Actual LGD Distribution")
    axes[0].set_xlabel("LGD")
    axes[0].set_ylabel("Density")

    axes[1].hist(y_pred, bins=50, alpha=0.7, color="#ef4444", edgecolor="white", density=True)
    axes[1].set_title("Predicted LGD Distribution")
    axes[1].set_xlabel("LGD")
    axes[1].set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(LGD_OUTPUT_DIR / "lgd_distributions.png")
    plt.close(fig)
    print(f"  Saved: {LGD_OUTPUT_DIR / 'lgd_distributions.png'}")

    # 3. Residual distribution
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residuals, bins=50, alpha=0.7, color="#8b5cf6", edgecolor="white", density=True)
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("LGD Residual Distribution")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(LGD_OUTPUT_DIR / "lgd_residuals.png")
    plt.close(fig)
    print(f"  Saved: {LGD_OUTPUT_DIR / 'lgd_residuals.png'}")

    # 4. LGD by decile
    df_eval = pd.DataFrame({"actual": y_test, "predicted": y_pred})
    df_eval["decile"] = pd.qcut(df_eval["predicted"], q=10, labels=False, duplicates="drop") + 1
    decile_stats = df_eval.groupby("decile").agg(
        actual_mean=("actual", "mean"),
        predicted_mean=("predicted", "mean"),
        count=("actual", "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(decile_stats))
    width = 0.35
    ax.bar(x - width / 2, decile_stats["actual_mean"], width, label="Actual", color="#3b82f6", edgecolor="white")
    ax.bar(x + width / 2, decile_stats["predicted_mean"], width, label="Predicted", color="#ef4444", edgecolor="white")
    ax.set_xlabel("Predicted LGD Decile (1=lowest → 10=highest)")
    ax.set_ylabel("Mean LGD")
    ax.set_title("Actual vs Predicted LGD by Decile")
    ax.set_xticks(x)
    ax.set_xticklabels(decile_stats["decile"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(LGD_OUTPUT_DIR / "lgd_by_decile.png")
    plt.close(fig)
    print(f"  Saved: {LGD_OUTPUT_DIR / 'lgd_by_decile.png'}")

    # 5. Feature importance (top 20)
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feat_imp["feature"], feat_imp["importance"], color="#22c55e", edgecolor="white")
    ax.set_title("LGD Model — Top 20 Feature Importances")
    ax.set_xlabel("Importance (gain)")
    fig.tight_layout()
    fig.savefig(LGD_OUTPUT_DIR / "lgd_feature_importance.png")
    plt.close(fig)
    print(f"  Saved: {LGD_OUTPUT_DIR / 'lgd_feature_importance.png'}")

    metrics = {"mae": mae, "rmse": rmse, "r2": r2,
               "mean_actual_lgd": y_test.mean(), "mean_predicted_lgd": y_pred.mean()}

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(LGD_OUTPUT_DIR / "lgd_metrics.csv", index=False)
    print(f"  Saved: {LGD_OUTPUT_DIR / 'lgd_metrics.csv'}")

    return metrics


def run_lgd_pipeline(n_tuning_trials: int = 15) -> dict:
    """Full LGD modelling pipeline."""
    train_def, test_def, features = load_defaulted_loans()

    X_train = train_def[features].values
    y_train = train_def["lgd"].values
    X_test = test_def[features].values
    y_test = test_def["lgd"].values

    best_params = tune_lgd_xgboost(X_train, y_train, n_trials=n_tuning_trials)
    model = train_lgd_model(X_train, y_train, best_params)
    metrics = evaluate_lgd_model(model, X_test, y_test, features)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "lgd_xgboost.joblib")
    joblib.dump(features, MODELS_DIR / "lgd_features.joblib")
    print(f"\nModel saved to {MODELS_DIR / 'lgd_xgboost.joblib'}")

    return metrics


if __name__ == "__main__":
    run_lgd_pipeline(n_tuning_trials=15)
