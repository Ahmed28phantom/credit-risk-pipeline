import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_PATH = project_root / "models" / "credit_risk_model.joblib"

NUMERIC_INPUTS = {
    "duration": {
        "label": "Loan duration (months)", "min": 4, "max": 72, "default": 18,
        "help": "Length of the loan term, in months.",
    },
    "amount": {
        "label": "Loan amount (DM)", "min": 250, "max": 18424, "default": 2320,
        "help": "Total credit amount requested, in Deutsche Mark (DM).",
    },
    "installment_rate": {
        "label": "Installment rate (% of income, 1-4)", "min": 1, "max": 4, "default": 3,
        "help": "Loan installment as a share of disposable income, banded 1 (lowest) to 4 (highest).",
    },
    "present_residence": {
        "label": "Years at current residence (1-4)", "min": 1, "max": 4, "default": 3,
        "help": "Years the applicant has lived at their current address.",
    },
    "age": {
        "label": "Age (years)", "min": 19, "max": 75, "default": 33,
        "help": "Applicant's age in years.",
    },
    "number_credits": {
        "label": "Existing credits at this bank", "min": 1, "max": 4, "default": 1,
        "help": "Number of existing credits the applicant already holds at this bank.",
    },
    "people_liable": {
        "label": "Number of dependants", "min": 1, "max": 2, "default": 1,
        "help": "Number of dependents the applicant is financially responsible for.",
    },
}

CATEGORY_INFO = {
    "status": {
        "label": "Checking account status",
        "help": "Balance status of the applicant's existing checking account.",
        "options": ["< 0 DM", "0–200 DM", ">= 200 DM", "no account"],
    },
    "credit_history": {
        "label": "Credit history",
        "help": "Applicant's track record repaying past credit.",
        "options": ["no credits", "all paid", "existing paid", "delay in past", "critical account"],
    },
    "purpose": {
        "label": "Loan purpose",
        "help": "What the loan will be used for.",
        "options": [
            "car (new)", "car (used)", "furniture", "TV/radio", "appliances",
            "repairs", "education", "vacation", "retraining", "business", "other",
        ],
    },
    "savings": {
        "label": "Savings account / bonds",
        "help": "Balance held in savings accounts or bonds.",
        "options": ["< 100 DM", "100–500 DM", "500–1000 DM", ">= 1000 DM", "unknown/none"],
    },
    "employment_duration": {
        "label": "Employment duration",
        "help": "How long the applicant has been with their current employer.",
        "options": ["unemployed", "< 1 yr", "1–4 yrs", "4–7 yrs", ">= 7 yrs"],
    },
    "personal_status_sex": {
        "label": "Personal status / sex",
        "help": "Applicant's sex combined with marital status.",
        "options": [
            "male divorced", "female divorced/married", "male single", "male married", "female single",
        ],
    },
    "other_debtors": {
        "label": "Other debtors / guarantors",
        "help": "Whether a co-applicant or guarantor is attached to this loan.",
        "options": ["none", "co-applicant", "guarantor"],
    },
    "property": {
        "label": "Property",
        "help": "Most valuable property the applicant owns.",
        "options": ["real estate", "savings/insurance", "car/other", "unknown/none"],
    },
    "other_installment_plans": {
        "label": "Other installment plans",
        "help": "Other installment debt held elsewhere (banks or retail stores).",
        "options": ["bank", "stores", "none"],
    },
    "housing": {
        "label": "Housing",
        "help": "Applicant's housing situation: rents, owns, or lives rent-free.",
        "options": ["rent", "own", "free"],
    },
    "job": {
        "label": "Job",
        "help": "Applicant's employment skill level and type.",
        "options": ["unskilled non-resident", "unskilled resident", "skilled", "highly skilled"],
    },
    "telephone": {
        "label": "Telephone",
        "help": "Whether the applicant has a phone registered in their own name.",
        "options": ["none", "yes"],
    },
    "foreign_worker": {
        "label": "Foreign worker",
        "help": "Whether the applicant is classified as a foreign worker.",
        "options": ["yes", "no"],
    },
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Credit Risk Predictor")
st.title("Credit Risk Predictor")
st.caption("Estimate default risk for a loan applicant using the tuned Logistic Regression model.")
st.caption(
    "Based on the 1994 UCI Statlog German Credit dataset — fields like DM currency, "
    "\"foreign worker,\" and landline telephone registration reflect that era and context "
    "rather than universal risk factors."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found at {MODEL_PATH}. "
        "Run notebooks/03_modeling.ipynb to train and save the model first."
    )
    st.stop()

model = load_model()

with st.form("credit_risk_form"):
    numeric_values = {}
    categorical_values = {}

    st.subheader("Numeric features")
    numeric_cols = st.columns(3)
    for i, feature in enumerate(NUMERIC_FEATURES):
        cfg = NUMERIC_INPUTS[feature]
        with numeric_cols[i % 3]:
            numeric_values[feature] = st.number_input(
                cfg["label"], min_value=cfg["min"], max_value=cfg["max"], value=cfg["default"],
            )
            st.caption(cfg["help"])

    st.subheader("Categorical features")
    categorical_cols = st.columns(3)
    for i, feature in enumerate(CATEGORICAL_FEATURES):
        cfg = CATEGORY_INFO[feature]
        with categorical_cols[i % 3]:
            categorical_values[feature] = st.selectbox(cfg["label"], cfg["options"])
            st.caption(cfg["help"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_row = {**numeric_values, **categorical_values}
    input_df = pd.DataFrame([input_row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    proba_bad = model.predict_proba(input_df)[0, 1]
    prediction = model.predict(input_df)[0]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"Predicted: Bad credit risk (probability of default: {proba_bad:.1%})")
    else:
        st.success(f"Predicted: Good credit risk (probability of default: {proba_bad:.1%})")
