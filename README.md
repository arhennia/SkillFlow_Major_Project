# Customer Churn Prediction & Intelligent Retention System

> **My Machine Learning Internship — Major Project**  
> IBM Telco Customer Churn Dataset · End-to-End ML Pipeline

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Deliverables](#deliverables)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Setup & Installation](#setup--installation)
6. [How to Run (Step-by-Step)](#how-to-run-step-by-step)
7. [Workflow Summary](#workflow-summary)
8. [Model Performance](#model-performance)
9. [Key Business Insights](#key-business-insights)
10. [Retention Strategies](#retention-strategies)
11. [File Descriptions](#file-descriptions)

---

## Project Overview

I built this complete machine learning pipeline to **predict customer churn** for a telecom company. 

**The problem:** About 26.5% of customers churn, which means a huge loss in recurring revenue. Trying to keep them after they've already decided to leave is just too late.

**The solution:** I developed a supervised ML model that assigns a churn probability score to every customer. This helps the business step in early with targeted retention offers.

**Dataset Used:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers and 21 features covering demographics, billing, and usage.

---

## Deliverables

| # | Deliverable | Location |
|---|---|---|
| 1 | Jupyter Notebook (fully executed, with all outputs) | `notebooks/Customer_Churn_Prediction.ipynb` |
| 2 | Business & Technical Report (PDF) | `reports/Business_and_Technical_Report.pdf` |
| 3 | Presentation (7 slides, PPTX) | `presentation/Churn_Prediction_Presentation.pptx` |
| 4 | Trained Model (`.pkl`) | `models/churn_model.pkl` |
| 5 | Fitted Scaler (`.pkl`) | `models/scaler.pkl` |
| 6 | Prediction Script | `src/predict.py` |
| 7 | Requirements File | `requirements.txt` |

---

## Project Structure

```
SkillFlow_Major_Project/
│
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── .gitignore
├── LICENSE
│
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb   ← Main notebook (Steps 1–7)
│
├── reports/
│   └── Business_and_Technical_Report.pdf ← Business & Technical Report (Step 8)
│
├── presentation/
│   └── Churn_Prediction_Presentation.pptx ← 7-slide deck (optional bonus)
│
├── models/                            ← All trained model artifacts
│   ├── churn_model.pkl                ← Champion model (fitted)
│   ├── scaler.pkl                     ← Fitted StandardScaler
│   ├── model_columns.pkl              ← Exact feature column order
│   ├── numeric_columns.pkl            ← Columns that were scaled
│   ├── model_comparison.csv           ← All model metrics
│   ├── top10_feature_importance.csv   ← Top 10 features
│   └── champion_model_info.json       ← Champion name + metrics + timestamp
│
├── src/                               ← All Python source scripts
│   ├── churn_utils.py                 ← Shared cleaning & feature engineering
│   ├── train_and_export.py            ← Full training pipeline
│   ├── predict.py                     ← Inference / prediction script
│   ├── build_notebook.py              ← Builds + executes the .ipynb
│   ├── generate_pdf_report.py         ← Generates the PDF report
│   └── generate_presentation.py      ← Generates the PPTX deck
│
└── data/
    ├── README.md                      ← Dataset info & download instructions
    └── Telco-Customer-Churn.csv       ← ⚠ Download separately (see data/README.md)
```

---

## Dataset

The **IBM Telco Customer Churn** dataset is **not included** in this repository (it must be downloaded separately).

**Download from Kaggle:**  
👉 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Or via curl:**
```bash
curl -o data/Telco-Customer-Churn.csv \
  "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
```

Place the file at `data/Telco-Customer-Churn.csv`. All scripts resolve paths automatically.

See [`data/README.md`](data/README.md) for a full column reference.

---

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- `pip`

### Install dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
joblib>=1.3
matplotlib>=3.7
seaborn>=0.12
reportlab>=4.0
python-pptx>=0.6.23
nbformat>=5.9
nbclient          # for executing the notebook programmatically
Pillow            # required by generate_presentation.py for image sizing
ipykernel         # required to execute the notebook
```

---

## How to Run (Step-by-Step)

> **All commands should be run from the project root** (`SkillFlow_Major_Project/`), not from inside `src/`.

### Step 1 — Download the dataset

Follow the instructions in [`data/README.md`](data/README.md) and place `Telco-Customer-Churn.csv` in the `data/` folder.

---

### Step 2 — Train all models & export artifacts

```bash
python src/train_and_export.py
```

This will:
- Load and clean the dataset
- Engineer 4 new features
- Train 5 models (Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM)
- Tune Random Forest and XGBoost with `RandomizedSearchCV` (5-fold Stratified CV)
- Select the champion model by ROC-AUC
- Save all artifacts to `models/`

Expected runtime: **5–15 minutes** depending on hardware.

---

### Step 3 — Build & execute the Jupyter Notebook

```bash
python src/build_notebook.py
```

This programmatically assembles and **fully executes** the notebook, baking all outputs and plots into `notebooks/Customer_Churn_Prediction.ipynb`.

You can then open it with:
```bash
jupyter notebook notebooks/Customer_Churn_Prediction.ipynb
```

---

### Step 4 — Generate the PDF Report

```bash
python src/generate_pdf_report.py
```

Reads model metrics from `models/` and produces `reports/Business_and_Technical_Report.pdf`.

> ⚠ Run **after** Step 2 (requires `model_comparison.csv`, `champion_model_info.json`, `top10_feature_importance.csv` in `models/`).

---

### Step 5 — Generate the Presentation (optional)

```bash
python src/generate_presentation.py
```

Produces `presentation/Churn_Prediction_Presentation.pptx`. Embeds the plot images from `notebooks/` if they exist.

> ⚠ Run **after** Step 3 (requires plot PNGs in `notebooks/`).

---

### Step 6 — Run inference on sample customers

```bash
python src/predict.py
```

Loads the trained model from `models/`, runs predictions on 5 hand-crafted sample customers, prints a risk-tiered output table, and saves `models/sample_predictions.csv`.

---

## Workflow Summary

```
Data                         Training                      Outputs
──────────────────────────── ───────────────────────────── ──────────────────────────────
data/                        src/train_and_export.py       models/churn_model.pkl
  Telco-Customer-Churn.csv   ├─ clean_data()               models/scaler.pkl
                             ├─ engineer_features()        models/model_comparison.csv
                             ├─ encode + scale             models/champion_model_info.json
                             ├─ train 5 models             models/top10_feature_importance.csv
                             └─ tune RF + XGBoost
                                                           notebooks/
                             src/build_notebook.py    →      Customer_Churn_Prediction.ipynb
                                                              plot_01_*.png ... plot_11_*.png

                             src/generate_pdf_report.py →  reports/
                                                              Business_and_Technical_Report.pdf

                             src/generate_presentation.py → presentation/
                                                              Churn_Prediction_Presentation.pptx

                             src/predict.py           →    models/sample_predictions.csv
```

---

## Model Performance

All models are trained on an **80/20 stratified train-test split** (`random_state=42`). The champion is selected by **ROC-AUC** (tie-break: **Recall**).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | ~0.80 | ~0.65 | ~0.54 | ~0.59 | ~0.84 |
| Decision Tree | ~0.78 | ~0.57 | ~0.50 | ~0.53 | ~0.72 |
| Random Forest (Default) | ~0.80 | ~0.66 | ~0.48 | ~0.56 | ~0.84 |
| XGBoost (Default) | ~0.81 | ~0.67 | ~0.54 | ~0.60 | ~0.85 |
| SVM (RBF) | ~0.80 | ~0.65 | ~0.55 | ~0.60 | ~0.85 |
| **Random Forest (Tuned)** | ~0.81 | ~0.67 | ~0.55 | ~0.61 | **~0.86** |
| **XGBoost (Tuned)** | ~0.81 | ~0.67 | ~0.56 | ~0.61 | **~0.86** |

> Exact metrics are stored in `models/model_comparison.csv` after running Step 2.

**Why ROC-AUC and Recall?**  
A missed churner (False Negative) costs the business the customer's full lifetime value plus replacement acquisition cost. A false alarm (False Positive) costs only a minor unnecessary retention offer. This asymmetry means **catching as many true churners as possible (Recall)** matters more than raw Accuracy, and **ROC-AUC** gives a threshold-independent view of ranking quality.

---

## Key Business Insights

1. **Contract type is the #1 churn driver.** Month-to-month customers churn at dramatically higher rates (~43%) versus one-year (~11%) or two-year (~3%) contracts.
2. **Churn is highest in the first 12 months.** The critical retention window is the onboarding period — after year one, churn risk drops substantially.
3. **Electronic check payers churn more.** Manual payment correlates with lower engagement and commitment compared to automatic payment methods.
4. **Fiber optic customers churn more than DSL users** — despite higher monthly charges — signalling a possible price/value or service perception gap.
5. **Customers without add-on services churn more.** Online Security, Tech Support, and similar add-ons increase stickiness and switching costs.
6. **High charges + short tenure + month-to-month = highest risk.** This combination defines the single most at-risk customer segment.

---

## Retention Strategies

| # | Strategy | Rationale |
|---|---|---|
| 1 | **Personalized Retention Offers** | Route 70%+ probability customers to tailored discounts/upgrades |
| 2 | **Loyalty & Contract Migration Rewards** | Incentivize M2M customers to upgrade to 1- or 2-year contracts |
| 3 | **Early Churn Alert System** | Monthly model scoring → automated CRM alerts for medium/high-risk customers |
| 4 | **Onboarding & Engagement Campaigns** | Structured 12-month onboarding journey for new customers |
| 5 | **Service Bundling Incentives** | Discounted add-on bundles to increase switching costs and perceived value |

---

## File Descriptions

| File | Description |
|---|---|
| `src/churn_utils.py` | Shared functions: `clean_data()`, `engineer_features()`, `build_model_matrix()`. Imported by all other scripts to guarantee identical preprocessing at train and inference time. |
| `src/train_and_export.py` | Full training pipeline. Run this first to produce all model artifacts. |
| `src/predict.py` | Standalone inference script. Loads artifacts from `models/` and scores raw customer records. |
| `src/build_notebook.py` | Programmatically builds and executes the Jupyter notebook using `nbformat` + `nbclient`. |
| `src/generate_pdf_report.py` | Generates the 3-page executive PDF using `reportlab`. Reads live model metrics — no hard-coded numbers. |
| `src/generate_presentation.py` | Generates the 7-slide PPTX using `python-pptx`. Embeds EDA and ROC curve plots. |
| `models/churn_model.pkl` | The champion fitted model (serialized with `joblib`). |
| `models/scaler.pkl` | The fitted `StandardScaler` (must be used at inference — never refit on new data). |
| `models/model_columns.pkl` | Ordered list of feature column names. Used to align inference data to training schema. |
| `data/README.md` | Dataset documentation and download instructions. |

---

## Notes

- All random seeds are set to `42` for reproducibility.
- The `churn_utils.py` module is shared across training and inference to prevent **train/serve skew** — the single most common production ML bug.
- The scaler is fitted **only on the training set** and applied (without refitting) to the test set and any new inference data.
- Running `train_and_export.py` will **overwrite** existing model artifacts in `models/`.

---

*Built as part of the Machine Learning Internship Major Project.*
