"""
churn_utils.py
================
Shared data-cleaning and feature-engineering utilities for the
Customer Churn Prediction & Intelligent Retention System project.

Importing this module guarantees that the notebook (EDA/training),
train_and_export.py, and predict.py all apply IDENTICAL transformations
to the raw IBM Telco Customer Churn data, which is critical for a
production ML pipeline (train/serve skew prevention).

Expected project layout (relative to this file in src/):
    ../data/Telco-Customer-Churn.csv   ← raw dataset
    ../models/                         ← trained model artifacts
"""

import os

import pandas as pd
import numpy as np

# Resolve paths relative to this file so scripts work regardless of the
# working directory from which they are invoked.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

RAW_CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

# Columns that are Yes/No add-on services used for Total_Services_Used
ADDON_SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    
    
]

# Binary Yes/No columns (excluding target) that get simple 0/1 mapping
BINARY_COLS = ["Partner", "Dependents", "PaperlessBilling"]

# Nominal categorical columns that get one-hot encoded
NOMINAL_COLS = [
    "gender", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaymentMethod", "PhoneService",
]

# Continuous numeric columns that get standard-scaled
NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "Total_Services_Used", "Estimated_LTV",
]


def load_raw_data(path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the raw IBM Telco Customer Churn CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data-hygiene pass:
      - TotalCharges arrives as an object dtype containing literal
        blank-space strings (" ") for the 11 customers with tenure == 0
        (brand-new customers who haven't been billed yet). Coerce to
        numeric and impute with 0 (they have not been charged anything).
      - Drop exact duplicate rows (none expected, but defensive).
      - Drop customerID (not predictive, purely an identifier).
    """
    df = df.copy()

    # --- TotalCharges: blank-string -> NaN -> numeric -> impute ---
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Customers with blank TotalCharges are all tenure == 0 (new signups);
    # their true lifetime spend to date is 0.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --- Duplicates ---
    df = df.drop_duplicates()

    # --- Drop identifier column if present (keep aside if needed later) ---
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the four required engineered features:
      1. Total_Services_Used
      2. Estimated_LTV
      3. Tenure_Group
      4. Payment_Risk_Score
    """
    df = df.copy()

    # 1. Total_Services_Used: count of subscribed add-on services
    df["Total_Services_Used"] = (df[ADDON_SERVICE_COLS] == "Yes").sum(axis=1)

    # 2. Estimated_LTV: MonthlyCharges * tenure
    df["Estimated_LTV"] = df["MonthlyCharges"] * df["tenure"]

    # 3. Tenure_Group: categorical binning
    bins = [-1, 12, 24, 48, np.inf]
    labels = ["0-12 Months", "12-24 Months", "24-48 Months", "48+ Months"]
    df["Tenure_Group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # 4. Payment_Risk_Score: 1 = high risk (Electronic check / manual),
    #    0 = low risk (automatic bank transfer / automatic credit card)
    automatic_methods = {
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    }
    df["Payment_Risk_Score"] = df["PaymentMethod"].apply(
        lambda x: 0 if x in automatic_methods else 1
    )

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binary-encode the Churn target: Yes -> 1, No -> 0."""
    df = df.copy()
    if "Churn" in df.columns and not pd.api.types.is_numeric_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return df


def full_clean_and_engineer(path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """Convenience wrapper: load -> clean -> engineer -> encode target."""
    df = load_raw_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_target(df)
    return df


def build_model_matrix(df: pd.DataFrame):
    """
    Given the cleaned + engineered dataframe (with Churn already 0/1),
    build the final X (feature matrix, pre-scaling/pre-encoding) and y.

    Encoding strategy:
      - SeniorCitizen is already 0/1 in the raw data.
      - BINARY_COLS: Yes/No -> 1/0
      - NOMINAL_COLS: one-hot encoded (drop_first=True to avoid
        multicollinearity in linear models)
      - Tenure_Group: one-hot encoded (ordinal info retained via tenure itself)
      - NUMERIC_COLS: left as-is here; scaling is applied separately with
        a fitted StandardScaler so train and inference share one scaler.
    """
    df = df.copy()
    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])

    # Binary Yes/No columns
    for col in BINARY_COLS:
        X[col] = X[col].map({"Yes": 1, "No": 0})

    # One-hot encode nominal + Tenure_Group
    onehot_cols = NOMINAL_COLS + ["Tenure_Group"]
    X = pd.get_dummies(X, columns=onehot_cols, drop_first=True)

    # Ensure all-numeric (SeniorCitizen, Payment_Risk_Score already numeric)
    X = X.astype({c: "float64" for c in X.select_dtypes(include=["bool"]).columns})

    return X, y
