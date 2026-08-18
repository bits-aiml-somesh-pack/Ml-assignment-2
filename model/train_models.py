"""
Train Models - ML Assignment 2
================================
Trains 5 classification models on the UCI Bank Marketing dataset
(bank-additional-full.csv) to predict whether a client will subscribe
to a term deposit (target column `y`).

Models trained:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbor Classifier
    4. Naive Bayes Classifier (Gaussian)
    5. Random Forest Classifier (Ensemble)

For each model, the following metrics are computed on the held-out test set:
    Accuracy, AUC, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC)

Outputs (written to the `model/` directory unless noted):
    - preprocessor.pkl                 (fitted ColumnTransformer)
    - logistic_regression.pkl
    - decision_tree.pkl
    - knn.pkl
    - naive_bayes.pkl
    - random_forest.pkl
    - metrics.json                     (comparison table data)
    - ../test_data.csv                 (held-out test split, raw features + target)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "bank-additional-full.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

RANDOM_STATE = 42

# Column `duration` is dropped: it is only known AFTER a call ends, so it
# leaks the outcome and is not available at prediction time in a real
# deployment. This is a well documented caveat for this dataset.
DROP_COLUMNS = ["duration"]

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
NUMERIC_COLS = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
TARGET_COL = "y"


def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")
    df = df.drop(columns=DROP_COLUMNS)
    return df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
    }


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()
    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)  # "no" -> 0, "yes" -> 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    results = {}
    filenames = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "kNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest (Ensemble)": "random_forest.pkl",
    }

    for name, model in get_models().items():
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_t)[:, 1]
        else:
            y_proba = model.decision_function(X_test_t)

        metrics = compute_metrics(y_test, y_pred, y_proba)
        results[name] = metrics

        joblib.dump(model, os.path.join(MODEL_DIR, filenames[name]))

        print(f"{name}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Save the raw (untransformed) test split for the Streamlit app.
    test_df = X_test.copy()
    test_df[TARGET_COL] = label_encoder.inverse_transform(y_test)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"\nSaved models and preprocessor to: {MODEL_DIR}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved test data ({len(test_df)} rows) to: {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
