import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "loan_approval_pipeline.pkl"
pipeline = joblib.load(MODEL_PATH)

EMPLOYMENT_STATUS_OPTIONS = ["Salaried", "Self-employed", "Contract", "Unemployed"]
MARITAL_STATUS_OPTIONS = ["Married", "Single"]
LOAN_PURPOSE_OPTIONS = ["Personal", "Car", "Business", "Home", "Education"]
PROPERTY_AREA_OPTIONS = ["Urban", "Semiurban", "Rural"]
GENDER_OPTIONS = ["Male", "Female"]
EMPLOYER_CATEGORY_OPTIONS = ["Private", "Government", "MNC", "Business", "Unemployed"]
EDUCATION_LEVEL_OPTIONS = ["Graduate", "Not Graduate"]

FORM_OPTIONS = {
    "Employment_Status": EMPLOYMENT_STATUS_OPTIONS,
    "Marital_Status": MARITAL_STATUS_OPTIONS,
    "Loan_Purpose": LOAN_PURPOSE_OPTIONS,
    "Property_Area": PROPERTY_AREA_OPTIONS,
    "Gender": GENDER_OPTIONS,
    "Employer_Category": EMPLOYER_CATEGORY_OPTIONS,
    "Education_Level": EDUCATION_LEVEL_OPTIONS,
}

# field name -> (label, input type, min, max, step, default)
NUMERIC_FIELDS = [
    ("Applicant_Income", "Applicant income (monthly)", 2000, 200000, 100, 9400),
    ("Coapplicant_Income", "Co-applicant income (monthly)", 0, 100000, 100, 8173),
    ("Age", "Age", 18, 70, 1, 42),
    ("Dependents", "Dependents", 0, 6, 1, 0),
    ("Credit_Score", "Credit score", 300, 900, 1, 780),
    ("Existing_Loans", "Existing loans", 0, 10, 1, 0),
    ("DTI_Ratio", "Debt-to-income ratio", 0.0, 1.0, 0.01, 0.22),
    ("Savings", "Savings", 0, 20000, 100, 558),
    ("Collateral_Value", "Collateral value", 0, 50000, 100, 13939),
    ("Loan_Amount", "Loan amount requested", 500, 50000, 100, 21976),
    ("Loan_Term", "Loan term (months)", 6, 120, 1, 12),
]


def build_customer_dataframe(form) -> pd.DataFrame:
    """Create a single-row DataFrame matching the training data."""

    data = {"Applicant_ID": 0}

    for field, *_ in NUMERIC_FIELDS:
        value = form.get(field)

        if field == "DTI_Ratio":
            data[field] = float(value)
        else:
            data[field] = int(float(value))

    data["Education_Level"] = (
        1 if form.get("Education_Level") == "Graduate" else 0
    )

    data["Employment_Status"] = form.get("Employment_Status")
    data["Marital_Status"] = form.get("Marital_Status")
    data["Loan_Purpose"] = form.get("Loan_Purpose")
    data["Property_Area"] = form.get("Property_Area")
    data["Gender"] = form.get("Gender")
    data["Employer_Category"] = form.get("Employer_Category")

    customer = pd.DataFrame([data])

    # Feature Engineering (must match train_model.py)
    customer["Applicant_Income_log"] = np.log1p(customer["Applicant_Income"])
    customer["Credit_Score_sq"] = customer["Credit_Score"] ** 2
    customer["DTI_Ratio_sq"] = customer["DTI_Ratio"] ** 2

    return customer


@app.route("/")
def index():
    return render_template(
        "index.html",
        numeric_fields=NUMERIC_FIELDS,
        options=FORM_OPTIONS,
        result=None,
    )


@app.route("/predict", methods=["POST"])
def predict():

    customer = build_customer_dataframe(request.form)

    prediction = pipeline.predict(customer)[0]
    probabilities = pipeline.predict_proba(customer)[0]

    result = {
        "approved": prediction == 1,
        "approve_prob": round(probabilities[1] * 100, 2),
        "reject_prob": round(probabilities[0] * 100, 2),
    }

    return render_template(
        "index.html",
        numeric_fields=NUMERIC_FIELDS,
        options=FORM_OPTIONS,
        result=result,
        form_values=request.form,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)