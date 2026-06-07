# Credit Risk Scorecard — PD Model

A production-style Probability of Default (PD) model built on the Home Credit Default Risk dataset. Combines credit risk domain knowledge with machine learning to predict loan defaults and generate explainable risk scores.

## Business Problem
A retail lending division needs to move beyond rule-based credit decisions. This project builds a statistically validated PD model that scores applicants on default probability, identifies key risk drivers, and generates an explainable scorecard for underwriting teams.

## Project Structure
- `notebooks/` — step-by-step analysis pipeline
- `data/processed/` — cleaned and feature-engineered datasets
- `reports/` — model evaluation charts and SHAP explanations
- `dashboard/` — interactive Streamlit risk analytics app
- `src/` — reusable Python modules

## Pipeline
| Phase | Notebook | Description |
|-------|----------|-------------|
| 1 | 01_eda.ipynb | Data loading, cleaning, null handling |
| 2 | 02_feature_engineering.ipynb | DTI ratio, bureau scores, risk tiers |
| 3 | 03_model_development.ipynb | Logistic Regression vs XGBoost |
| 4 | 04_explainability.ipynb | SHAP global and individual explanations |

## Model Results
| Model | AUROC | Gini | KS |
|-------|-------|------|----|
| Logistic Regression | 0.746 | 0.492 | 0.365 |
| XGBoost | 0.762 | 0.523 | 0.386 |

## Key Risk Findings
- External bureau score is the dominant default predictor (3x more important than any other feature)
- Borrowers under 25 default at 12.3% vs 5.2% for borrowers over 55
- Transport, Construction and Restaurant sector employees show 2x higher default rates than average
- Thin-file borrowers (missing bureau data) are significantly higher risk

## Tech Stack
Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, SQL

## FRM Concepts Applied
Expected Loss framework, Basel II IRB approach, IFRS 9 ECL staging, Model Risk (SR 11-7), Population Stability Index

## Dataset
Home Credit Default Risk — 307,511 loan applications, 122 features
