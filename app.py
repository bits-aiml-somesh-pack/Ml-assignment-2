"""
Streamlit App - ML Assignment 2
================================
Interactive demo for comparing 5 classification models trained on the
UCI Bank Marketing dataset (term deposit subscription prediction).

Features:
    - Upload a test CSV (raw feature columns + `y` target column)
    - Select which trained model to evaluate from a dropdown
    - View evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
    - View confusion matrix and classification report for the selection
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

TARGET_COL = "y"

st.set_page_config(page_title="Bank Marketing Classifier Comparison", layout="wide")


@st.cache_resource
def load_preprocessor_and_encoder():
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return preprocessor, label_encoder


@st.cache_resource
def load_model(model_filename):
    return joblib.load(os.path.join(MODEL_DIR, model_filename))


@st.cache_data
def load_training_metrics():
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return {}


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
    st.title("Bank Marketing — Classification Model Comparison")
    st.markdown(
        "Predicting whether a client will subscribe to a **term deposit** "
        "(UCI Bank Marketing dataset). Upload the provided `test_data.csv`, "
        "pick a model, and review its performance."
    )

    st.sidebar.header("Controls")
    model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

    uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])

    training_metrics = load_training_metrics()

    st.subheader("Model Comparison (from training run)")
    if training_metrics:
        comparison_df = pd.DataFrame(training_metrics).T
        comparison_df = comparison_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
        st.dataframe(comparison_df.style.format("{:.4f}"), use_container_width=True)
    else:
        st.info("No training metrics found. Run `model/train_models.py` first.")

    st.divider()

    if uploaded_file is None:
        st.info("Upload a CSV file (e.g. `test_data.csv`) in the sidebar to evaluate a model on it.")
        return

    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded CSV file: {e}")
        return

    if TARGET_COL not in data.columns:
        st.error(f"Uploaded CSV must contain a '{TARGET_COL}' target column.")
        return

    preprocessor, label_encoder = load_preprocessor_and_encoder()
    model = load_model(MODEL_FILES[model_name])

    X = data.drop(columns=[TARGET_COL])
    y_true_raw = data[TARGET_COL]

    try:
        y_true = label_encoder.transform(y_true_raw)
    except ValueError as e:
        st.error(f"Target column contains unexpected labels: {e}")
        return

    try:
        X_t = preprocessor.transform(X)
    except Exception as e:
        st.error(
            "Could not transform the uploaded data with the fitted preprocessor. "
            f"Make sure the column names/types match the training data. Details: {e}"
        )
        return

    y_pred = model.predict(X_t)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_t)[:, 1]
    else:
        y_proba = model.decision_function(X_t)

    metrics = compute_metrics(y_true, y_pred, y_proba)

    st.subheader(f"Results on Uploaded Data — {model_name}")

    cols = st.columns(6)
    metric_labels = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for col, label in zip(cols, metric_labels):
        col.metric(label, f"{metrics[label]:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        st.markdown("**Classification Report**")
        report = classification_report(
            y_true, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    with st.expander("Preview of uploaded data"):
        st.dataframe(data.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
