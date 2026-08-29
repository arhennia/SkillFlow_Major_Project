# Dataset

## IBM Telco Customer Churn Dataset

This project uses the **IBM Sample Data — Telco Customer Churn** dataset.

| Property | Details |
|---|---|
| Rows | 7,043 customers |
| Columns | 21 attributes |
| Target variable | `Churn` (Yes / No) |
| Overall churn rate | ~26.5% |

### Attributes

| Column | Type | Description |
|---|---|---|
| `customerID` | String | Unique identifier (dropped before modelling) |
| `gender` | Categorical | Female / Male |
| `SeniorCitizen` | Binary (0/1) | Whether the customer is a senior citizen |
| `Partner` | Categorical | Yes / No |
| `Dependents` | Categorical | Yes / No |
| `tenure` | Integer | Months with the company |
| `PhoneService` | Categorical | Yes / No |
| `MultipleLines` | Categorical | Yes / No / No phone service |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Categorical | Yes / No / No internet service |
| `OnlineBackup` | Categorical | Yes / No / No internet service |
| `DeviceProtection` | Categorical | Yes / No / No internet service |
| `TechSupport` | Categorical | Yes / No / No internet service |
| `StreamingTV` | Categorical | Yes / No / No internet service |
| `StreamingMovies` | Categorical | Yes / No / No internet service |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Categorical | Yes / No |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic) |
| `MonthlyCharges` | Float | Current monthly charge ($) |
| `TotalCharges` | Float | Total amount charged (stored as string in raw CSV — see data hygiene notes) |
| `Churn` | Categorical → Binary | **Target variable** — Yes (churned) / No (retained) |

---

## How to Download

### Option A — Kaggle (recommended)

1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Click **Download** (requires a free Kaggle account)
3. Extract and rename the file to `Telco-Customer-Churn.csv`
4. Place it in **this `data/` folder**

### Option B — IBM Sample Data

Available directly from IBM's GitHub:
```
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

You can download it with:
```bash
curl -o data/Telco-Customer-Churn.csv \
  "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
```

---

## Why is this file not in the repository?

The dataset is not committed to git because:
- It is publicly available from the sources above
- Storing large data files in git inflates repository size
- The license terms of the dataset require attribution to the original source

Once you download the file, all scripts will work without any further configuration.
