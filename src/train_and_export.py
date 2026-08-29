"""
train_and_export.py
====================
Standalone production training script for the Customer Churn Prediction
& Intelligent Retention System.

What it does:
  1. Loads & cleans the IBM Telco Customer Churn dataset.
  2. Engineers the 4 required features.
  3. Splits data (80/20, stratified, random_state=42).
  4. Scales numeric features with StandardScaler.
  5. Trains a battery of models (Logistic Regression baseline, Decision
     Tree, Random Forest, XGBoost, SVM) and tunes Random Forest + XGBoost
     with StratifiedKFold RandomizedSearchCV optimizing ROC-AUC.
  6. Selects the champion model by ROC-AUC (tie-break: Recall).
  7. Exports:
       - models/churn_model.pkl   (best fitted model, via joblib)
       - models/scaler.pkl        (fitted StandardScaler, via joblib)
       - models/model_columns.pkl (exact column order expected by the model)
       - models/numeric_columns.pkl
       - models/model_comparison.csv  (metrics table for report/notebook reuse)
       - models/champion_model_info.json

Run (from project root):
    python src/train_and_export.py
"""

import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Ensure src/ is on the path when run from any directory
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

from churn_utils import NUMERIC_COLS, build_model_matrix, full_clean_and_engineer

MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def evaluate(model, X_test, y_test, model_name):
    """Compute the standard classification metrics for one fitted model."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION — TRAINING PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load, clean, engineer
    # ---------------------------------------------------------------
    print("\n[1/6] Loading & cleaning data...")
    df = full_clean_and_engineer()
    print(f"      Final shape after cleaning: {df.shape}")

    X, y = build_model_matrix(df)
    model_columns = X.columns.tolist()
    print(f"      Feature matrix shape: {X.shape}")

    # ---------------------------------------------------------------
    # 2. Train/test split
    # ---------------------------------------------------------------
    print("\n[2/6] Splitting data (80/20, stratified, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"      Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"      Train churn rate: {y_train.mean():.3f} | "
          f"Test churn rate: {y_test.mean():.3f}")

    # ---------------------------------------------------------------
    # 3. Scale numeric features (fit on train only!)
    # ---------------------------------------------------------------
    print("\n[3/6] Scaling continuous numeric features...")
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test_scaled[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    # ---------------------------------------------------------------
    # 4. Baseline + advanced models
    # ---------------------------------------------------------------
    print("\n[4/6] Training baseline & advanced models...")
    results = []
    fitted_models = {}

    baseline_lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    baseline_lr.fit(X_train_scaled, y_train)
    results.append(evaluate(baseline_lr, X_test_scaled, y_test, "Logistic Regression (Baseline)"))
    fitted_models["Logistic Regression (Baseline)"] = baseline_lr
    print("      - Logistic Regression trained.")

    dt = DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
    dt.fit(X_train_scaled, y_train)
    results.append(evaluate(dt, X_test_scaled, y_test, "Decision Tree"))
    fitted_models["Decision Tree"] = dt
    print("      - Decision Tree trained.")

    rf_default = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf_default.fit(X_train_scaled, y_train)
    results.append(evaluate(rf_default, X_test_scaled, y_test, "Random Forest (Default)"))
    fitted_models["Random Forest (Default)"] = rf_default
    print("      - Random Forest (default) trained.")

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE)
    gb.fit(X_train_scaled, y_train)
    results.append(evaluate(gb, X_test_scaled, y_test, "Gradient Boosting"))
    fitted_models["Gradient Boosting"] = gb
    print("      - Gradient Boosting trained.")

    xgb_default = XGBClassifier(
        n_estimators=200,
        eval_metric="logloss", random_state=RANDOM_STATE,
    )
    xgb_default.fit(X_train_scaled, y_train)
    results.append(evaluate(xgb_default, X_test_scaled, y_test, "XGBoost (Default)"))
    fitted_models["XGBoost (Default)"] = xgb_default
    print("      - XGBoost (default) trained.")

    svm = SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE)
    svm.fit(X_train_scaled, y_train)
    results.append(evaluate(svm, X_test_scaled, y_test, "SVM (RBF)"))
    fitted_models["SVM (RBF)"] = svm
    print("      - SVM trained.")

    # ---------------------------------------------------------------
    # 5. Hyperparameter tuning (GridSearchCV + RandomizedSearchCV)
    # ---------------------------------------------------------------
    print("\n[5/6] Hyperparameter tuning (GridSearchCV & RandomizedSearchCV, 5-fold Stratified CV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    dt_param_grid = {
        "max_depth": [3, 5, 6, 8, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    }
    dt_grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=dt_param_grid,
        scoring="roc_auc", cv=cv, n_jobs=-1,
    )
    dt_grid_search.fit(X_train_scaled, y_train)
    dt_grid_tuned = dt_grid_search.best_estimator_
    results.append(evaluate(dt_grid_tuned, X_test_scaled, y_test, "Decision Tree (GridSearchCV)"))
    fitted_models["Decision Tree (GridSearchCV)"] = dt_grid_tuned
    print(f"      - Decision Tree tuned (GridSearchCV). Best CV ROC-AUC: {dt_grid_search.best_score_:.4f}")
    print(f"        Best params: {dt_grid_search.best_params_}")

    rf_param_dist = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [5, 8, 10, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=rf_param_dist,
        n_iter=25, scoring="roc_auc", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    rf_search.fit(X_train_scaled, y_train)
    rf_tuned = rf_search.best_estimator_
    results.append(evaluate(rf_tuned, X_test_scaled, y_test, "Random Forest (Tuned)"))
    fitted_models["Random Forest (Tuned)"] = rf_tuned
    print(f"      - Random Forest tuned. Best CV ROC-AUC: {rf_search.best_score_:.4f}")
    print(f"        Best params: {rf_search.best_params_}")

    xgb_param_dist = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "scale_pos_weight": [1, 2, 2.77],  # ~class imbalance ratio
    }
    xgb_search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        param_distributions=xgb_param_dist,
        n_iter=25, scoring="roc_auc", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    xgb_search.fit(X_train_scaled, y_train)
    xgb_tuned = xgb_search.best_estimator_
    results.append(evaluate(xgb_tuned, X_test_scaled, y_test, "XGBoost (Tuned)"))
    fitted_models["XGBoost (Tuned)"] = xgb_tuned
    print(f"      - XGBoost tuned. Best CV ROC-AUC: {xgb_search.best_score_:.4f}")
    print(f"        Best params: {xgb_search.best_params_}")

    # ---------------------------------------------------------------
    # 6. Model comparison, champion selection, export
    # ---------------------------------------------------------------
    print("\n[6/6] Comparing models & exporting champion...")
    comparison_df = pd.DataFrame(results).sort_values(
        by=["ROC-AUC", "Recall"], ascending=False
    ).reset_index(drop=True)
    print("\n" + comparison_df.to_string(index=False))

    champion_name = comparison_df.iloc[0]["Model"]
    champion_model = fitted_models[champion_name]
    print(f"\n>>> CHAMPION MODEL: {champion_name} "
          f"(ROC-AUC={comparison_df.iloc[0]['ROC-AUC']}, "
          f"Recall={comparison_df.iloc[0]['Recall']})")

    # Persist artifacts to models/
    joblib.dump(champion_model, os.path.join(MODELS_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(model_columns, os.path.join(MODELS_DIR, "model_columns.pkl"))
    joblib.dump(NUMERIC_COLS, os.path.join(MODELS_DIR, "numeric_columns.pkl"))
    comparison_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)

    # Export top-10 feature importance for the PDF report
    if hasattr(champion_model, "feature_importances_"):
        importances = pd.Series(champion_model.feature_importances_, index=model_columns)
    else:
        importances = pd.Series(
            abs(champion_model.coef_[0]) if hasattr(champion_model, "coef_") else [],
            index=model_columns,
        )
    top10 = importances.sort_values(ascending=False).head(10)
    top10.reset_index().to_csv(
        os.path.join(MODELS_DIR, "top10_feature_importance.csv"), index=False
    )

    with open(os.path.join(MODELS_DIR, "champion_model_info.json"), "w") as f:
        json.dump(
            {
                "champion_model": champion_name,
                "metrics": comparison_df.iloc[0].to_dict(),
                "n_features": len(model_columns),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f, indent=2,
        )

    print("\nArtifacts written to models/:")
    for fname in [
        "churn_model.pkl", "scaler.pkl", "model_columns.pkl",
        "numeric_columns.pkl", "model_comparison.csv",
        "top10_feature_importance.csv", "champion_model_info.json",
    ]:
        print(f"      - {fname}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
