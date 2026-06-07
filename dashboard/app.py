import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("Credit Risk Scorecard — Portfolio Analytics")
st.markdown("Home Credit Default Risk | FRM Credit Risk Project")

@st.cache_data
def load_data():
    return pd.read_csv('../data/processed/application_train_features.csv')

@st.cache_resource
def train_models(df):
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg/pos, eval_metric='auc',
        random_state=42, verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, xgb_proba)
    gini = 2 * auroc - 1
    fpr, tpr, _ = roc_curve(y_test, xgb_proba)
    ks = max(tpr - fpr)
    return xgb, X_train, X_test, y_test, xgb_proba, auroc, gini, ks

df = load_data()
xgb, X_train, X_test, y_test, xgb_proba, auroc, gini, ks = train_models(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio Default Rate", f"{round(df['TARGET'].mean() * 100, 2)}%")
col2.metric("Total Applications", f"{len(df):,}")
col3.metric("Model AUROC", f"{round(auroc, 3)}")
col4.metric("Gini Coefficient", f"{round(gini, 3)}")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Score Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(xgb_proba[y_test == 0], bins=50, alpha=0.6, 
            label='Good borrower', color='steelblue')
    ax.hist(xgb_proba[y_test == 1], bins=50, alpha=0.6, 
            label='Defaulter', color='crimson')
    ax.set_xlabel('Default Probability')
    ax.set_ylabel('Count')
    ax.legend()
    st.pyplot(fig)

with col_right:
    st.subheader("ROC Curve")
    fpr_vals, tpr_vals, _ = roc_curve(y_test, xgb_proba)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(fpr_vals, tpr_vals, color='darkorange', 
             label=f'XGBoost (AUROC={round(auroc, 3)})')
    ax2.plot([0, 1], [0, 1], 'k--', label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    st.pyplot(fig2)

st.markdown("---")
st.subheader("Individual Application Scorer")
st.markdown("Adjust the sliders to score a new applicant")

col_a, col_b, col_c = st.columns(3)

with col_a:
    ext_source = st.slider("Bureau Credit Score (0-1)", 0.0, 1.0, 0.5)
    days_employed = st.slider("Days Employed", 0, 3000, 1000)

with col_b:
    credit_annuity = st.slider("Credit to Annuity Ratio", 5.0, 60.0, 20.0)
    age_years = st.slider("Age (years)", 20, 70, 35)

with col_c:
    credit_income = st.slider("Credit to Income Ratio", 0.5, 10.0, 3.0)
    annuity_income = st.slider("Annuity to Income Ratio", 0.0, 0.5, 0.18)

if st.button("Score This Applicant"):
    sample = X_test.iloc[[0]].copy()
    sample['EXT_SOURCE_MEAN'] = ext_source
    sample['EXT_SOURCE_MIN'] = ext_source * 0.9
    sample['DAYS_EMPLOYED'] = days_employed
    sample['CREDIT_ANNUITY_RATIO'] = credit_annuity
    sample['AGE_YEARS'] = age_years
    sample['CREDIT_INCOME_RATIO'] = credit_income
    sample['ANNUITY_INCOME_RATIO'] = annuity_income

    prob = xgb.predict_proba(sample)[0][1]
    
    st.markdown("---")
    if prob < 0.10:
        st.success(f"LOW RISK — Default Probability: {round(prob * 100, 2)}% → Recommend APPROVE")
    elif prob < 0.25:
        st.warning(f"MEDIUM RISK — Default Probability: {round(prob * 100, 2)}% → Recommend REVIEW")
    else:
        st.error(f"HIGH RISK — Default Probability: {round(prob * 100, 2)}% → Recommend DECLINE")

        