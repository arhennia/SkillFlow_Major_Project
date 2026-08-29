"""
predict.py
==========
Standalone inference script for the Customer Churn Prediction system.

Loads the artifacts produced by train_and_export.py from models/:
    - models/churn_model.pkl
    - models/scaler.pkl
    - models/model_columns.pkl
    - models/numeric_columns.pkl

and runs predictions on sample raw customer records (in the same
schema as the original IBM Telco Customer Churn CSV), printing a
probability score and a business-friendly risk flag for each.

Run (from project root):
    python src/predict.py

Output:
    Prints a prediction table to stdout.
    Saves detailed results to models/sample_predictions.csv
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

# Ensure src/ is importable regardless of working directory
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

from churn_utils import clean_data, engineer_features, BINARY_COLS, NOMINAL_COLS

MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_COLUMNS_PATH = os.path.join(MODELS_DIR, "model_columns.pkl")
NUMERIC_COLUMNS_PATH = os.path.join(MODELS_DIR, "numeric_columns.pkl")


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_columns = joblib.load(MODEL_COLUMNS_PATH)
    numeric_columns = joblib.load(NUMERIC_COLUMNS_PATH)
    return model, scaler, model_columns, numeric_columns


def preprocess_for_inference(raw_df, scaler, model_columns, numeric_columns):
    """
    Apply the exact same cleaning + feature engineering + encoding used
    at training time, then align columns to `model_columns` so the
    inference matrix has identical shape/order to what the model expects
    (any category unseen at inference simply yields a 0 in that dummy
    column; any dummy column not present in this batch is added as 0).
    """
    df = raw_df.copy()

    # customerID may or may not be present; clean_data drops it if present
    df = clean_data(df)
    df = engineer_features(df)

    # Binary Yes/No columns -> 0/1
    for col in BINARY_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # One-hot encode nominal + Tenure_Group (must match training encoding)
    onehot_cols = NOMINAL_COLS + ["Tenure_Group"]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    # Drop Churn if present (not needed for inference)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # Align to training columns: add missing dummy cols as 0, drop extras,
    # and enforce the exact training column order.
    df = df.reindex(columns=model_columns, fill_value=0)
    df = df.astype("float64")

    # Scale numeric features with the FITTED training scaler (no re-fitting!)
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    return df


def risk_flag(probability: float) -> str:
    """Business-friendly churn risk tier."""
    if probability >= 0.70:
        return "HIGH RISK"
    elif probability >= 0.40:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"


def build_sample_customers() -> pd.DataFrame:
    """
    A handful of hand-crafted sample customers spanning different risk
    profiles, in the exact raw schema of the original dataset (minus
    customerID/Churn, which aren't needed for inference).
    """
    samples = [
        # High-risk archetype: new customer, month-to-month, electronic check,
        # no add-ons, high monthly charge, fiber optic internet.
        dict(
            gender="Female", SeniorCitizen=0, Partner="No", Dependents="No",
            tenure=2, PhoneService="Yes", MultipleLines="No",
            InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
            DeviceProtection="No", TechSupport="No", StreamingTV="Yes",
            StreamingMovies="Yes", Contract="Month-to-month", PaperlessBilling="Yes",
            PaymentMethod="Electronic check", MonthlyCharges=95.50, TotalCharges=191.00,
        ),
        # Low-risk archetype: long-tenured, two-year contract, automatic
        # payment, several add-on services.
        dict(
            gender="Male", SeniorCitizen=0, Partner="Yes", Dependents="Yes",
            tenure=60, PhoneService="Yes", MultipleLines="Yes",
            InternetService="DSL", OnlineSecurity="Yes", OnlineBackup="Yes",
            DeviceProtection="Yes", TechSupport="Yes", StreamingTV="No",
            StreamingMovies="No", Contract="Two year", PaperlessBilling="No",
            PaymentMethod="Bank transfer (automatic)", MonthlyCharges=65.20,
            TotalCharges=3912.00,
        ),
        # Medium-risk archetype: mid-tenure, one-year contract, mailed check,
        # partial add-ons.
        dict(
            gender="Female", SeniorCitizen=1, Partner="No", Dependents="No",
            tenure=18, PhoneService="Yes", MultipleLines="No",
            InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="Yes",
            DeviceProtection="No", TechSupport="No", StreamingTV="Yes",
            StreamingMovies="No", Contract="One year", PaperlessBilling="Yes",
            PaymentMethod="Mailed check", MonthlyCharges=79.85, TotalCharges=1437.30,
        ),
        # High-risk archetype: senior citizen, no partner/dependents, fiber,
        # month-to-month, electronic check, minimal tenure.
        dict(
            gender="Male", SeniorCitizen=1, Partner="No", Dependents="No",
            tenure=4, PhoneService="Yes", MultipleLines="Yes",
            InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
            DeviceProtection="No", TechSupport="No", StreamingTV="No",
            StreamingMovies="No", Contract="Month-to-month", PaperlessBilling="Yes",
            PaymentMethod="Electronic check", MonthlyCharges=89.10, TotalCharges=356.40,
        ),
        # Low-risk archetype: no internet service at all (phone-only), long
        # tenure, automatic credit card.
        dict(
            gender="Female", SeniorCitizen=0, Partner="Yes", Dependents="No",
            tenure=45, PhoneService="Yes", MultipleLines="No",
            InternetService="No", OnlineSecurity="No internet service",
            OnlineBackup="No internet service", DeviceProtection="No internet service",
            TechSupport="No internet service", StreamingTV="No internet service",
            StreamingMovies="No internet service", Contract="Two year",
            PaperlessBilling="No", PaymentMethod="Credit card (automatic)",
            MonthlyCharges=25.35, TotalCharges=1140.75,
        ),
    ]
    return pd.DataFrame(samples)


def predict_customers(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: raw customer rows -> churn probability + risk flag."""
    model, scaler, model_columns, numeric_columns = load_artifacts()
    X = preprocess_for_inference(raw_df, scaler, model_columns, numeric_columns)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    output = raw_df.copy()
    output["Churn_Probability"] = np.round(probabilities, 4)
    output["Churn_Prediction"] = np.where(predictions == 1, "Yes", "No")
    output["Risk_Flag"] = [risk_flag(p) for p in probabilities]
    return output


def main():
    print("=" * 70)
    print("CUSTOMER CHURN — INFERENCE ON SAMPLE CUSTOMERS")
    print("=" * 70)

    samples = build_sample_customers()
    results = predict_customers(samples)

    display_cols = [
        "tenure", "Contract", "PaymentMethod", "MonthlyCharges",
        "Churn_Probability", "Churn_Prediction", "Risk_Flag",
    ]
    print("\n" + results[display_cols].to_string(index=True))

    print("\nSummary:")
    for tier in ["HIGH RISK", "MEDIUM RISK", "LOW RISK"]:
        n = (results["Risk_Flag"] == tier).sum()
        print(f"  {tier}: {n} customer(s)")

    out_path = os.path.join(MODELS_DIR, "sample_predictions.csv")
    results.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    main()
