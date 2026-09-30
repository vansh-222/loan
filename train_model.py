"""
Trains the CreditWise loan-approval pipeline and saves it to disk
as loan_approval_pipeline.pkl, ready for the Flask app to load.

This mirrors the (fixed) logic from testing_proj_fixed.ipynb:
- median/mode imputation (proper assignment, not the broken inplace=True)
- feature engineering (log income, squared credit score / DTI ratio)
- education level mapping
- target encoding + dropping rows with missing target
- ColumnTransformer (OneHotEncoder + StandardScaler) + LogisticRegression
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_PATH = "loan_approval_data.csv"
MODEL_PATH = "loan_approval_pipeline.pkl"

CATEGORICAL_COLUMNS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
]


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Impute missing values (assign back properly -- inplace=True silently
    # no-ops under pandas Copy-on-Write and was the original notebook's bug)
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Feature engineering
    df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2

    # Encode Education_Level
    df["Education_Level"] = df["Education_Level"].map(
        {"Not Graduate": 0, "Graduate": 1}
    )

    # Handle target: drop rows with missing target, encode Yes/No -> 1/0
    df = df.dropna(subset=["Loan_Approved"])
    df["Loan_Approved"] = df["Loan_Approved"].map({"No": 0, "Yes": 1})

    return df


def build_pipeline(numerical_columns: list) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
            ("num", StandardScaler(), numerical_columns),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )


def main():
    df = load_and_clean_data(DATA_PATH)

    X = df.drop(columns=["Loan_Approved"])
    y = df["Loan_Approved"]

    numerical_columns = [c for c in X.columns if c not in CATEGORICAL_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numerical_columns)
    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    print(f"Training Accuracy : {train_acc * 100:.2f}%")
    print(f"Testing Accuracy  : {test_acc * 100:.2f}%")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Pipeline saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
