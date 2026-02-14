import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(
    page_title="ML Model Classifier",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------
# Paths (assumes files are in repo root)
# ----------------------------
MODEL_FILES = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl",
    "XGBoost (Ensemble)": "model/xgboost.pkl",
}

PIPELINE_FILE = "model/preprocessing_pipeline.pkl"
COMPARISON_CSV = "model/model_comparison.csv"
COMPARISON_PNG = "model/model_comparison.png"

# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_FILE)

@st.cache_resource
def load_model(model_name: str):
    return joblib.load(MODEL_FILES[model_name])

def get_proba(model, X):
    """Return probability for positive class, if available."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Binary: use column 1 for positive class
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        # Convert decision scores to (0,1) via sigmoid
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return None

def ensure_feature_order(df: pd.DataFrame, reference_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in reference_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in reference_cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if extra:
        # allow extra columns, but drop them to avoid shape mismatch
        df = df.drop(columns=extra)
    return df[reference_cols]

def plot_confusion(cm, labels=("0", "1")):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.tight_layout()
    return fig


# ----------------------------
# UI
# ----------------------------
st.title("🤖 Multi-Model Classification Dashboard")
st.markdown(
    """
Upload your **test features CSV** and optionally the **test labels CSV** to compute metrics.
Select a model to run predictions.  
This app loads the pre-trained models from `.pkl` files and applies the saved preprocessing pipeline.
"""
)

with st.sidebar:
    st.header("⚙️ Configuration")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()), index=0)

    st.markdown("---")
    st.subheader("📤 Upload test data")
    x_file = st.file_uploader("Test features (CSV)", type=["csv"])
    y_file = st.file_uploader("Test labels (optional, CSV with one column)", type=["csv"])

    st.markdown("---")
    show_comparison = st.checkbox("Show model comparison chart/table", value=True)

# Show comparison (precomputed)
if show_comparison:
    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("📊 Model comparison (precomputed)")
        if os.path.exists(COMPARISON_CSV):
            comp = pd.read_csv(COMPARISON_CSV)
            st.dataframe(comp, use_container_width=True)
        else:
            st.info("model_comparison.csv not found in repo.")
    with colB:
        if os.path.exists(COMPARISON_PNG):
            st.image(COMPARISON_PNG, caption="Model Performance Comparison", use_container_width=True)
        else:
            st.info("model_comparison.png not found in repo.")

st.markdown("---")

if x_file is None:
    st.info("Upload a test features CSV to begin.")
    st.stop()

# Load data
X_raw = pd.read_csv(x_file)

# Reference columns: use the columns from uploaded test_data OR the saved pipeline if you stored them.
# Here we enforce ordering based on the *uploaded* file, and later (if labels) we align by index length.
reference_cols = list(X_raw.columns)

# Load model + pipeline
pipeline = load_pipeline()
model = load_model(model_name)

# Preprocess (pipeline is a StandardScaler in your files)
try:
    X = pipeline.transform(X_raw)
except Exception as e:
    st.error("Preprocessing failed. Make sure your uploaded CSV has the same feature columns and types as training.")
    st.exception(e)
    st.stop()

# Predict
try:
    y_pred = model.predict(X)
except Exception as e:
    st.error("Prediction failed. Check preprocessing and model compatibility.")
    st.exception(e)
    st.stop()

y_proba = get_proba(model, X)

# Display predictions
st.subheader("✅ Predictions")
pred_df = pd.DataFrame({
    "prediction": y_pred.astype(int) if np.issubdtype(np.array(y_pred).dtype, np.number) else y_pred
})
if y_proba is not None:
    pred_df["probability_positive"] = y_proba

st.dataframe(pred_df.head(50), use_container_width=True)

# Download predictions
csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download predictions as CSV",
    data=csv_bytes,
    file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
    mime="text/csv",
)

# Metrics if labels provided
if y_file is not None:
    y_true_df = pd.read_csv(y_file)
    # Accept first column as target
    y_true = y_true_df.iloc[:, 0].values

    # Safety: align lengths
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred_aligned = np.array(y_pred)[:n]
    y_proba_aligned = y_proba[:n] if y_proba is not None else None

    st.markdown("---")
    st.subheader("📈 Evaluation Metrics")

    # Compute metrics
    acc = accuracy_score(y_true, y_pred_aligned)
    prec = precision_score(y_true, y_pred_aligned, zero_division=0)
    rec = recall_score(y_true, y_pred_aligned, zero_division=0)
    f1 = f1_score(y_true, y_pred_aligned, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred_aligned)

    auc = None
    if y_proba_aligned is not None:
        try:
            auc = roc_auc_score(y_true, y_proba_aligned)
        except Exception:
            auc = None

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("AUC", f"{auc:.3f}" if auc is not None else "N/A")
    m3.metric("Precision", f"{prec:.3f}")
    m4.metric("Recall", f"{rec:.3f}")
    m5.metric("F1", f"{f1:.3f}")
    m6.metric("MCC", f"{mcc:.3f}")

    # Confusion matrix + report
    c1, c2 = st.columns([1, 1.2])
    with c1:
        cm = confusion_matrix(y_true, y_pred_aligned)
        st.pyplot(plot_confusion(cm))
    with c2:
        st.text("Classification Report")
        st.code(classification_report(y_true, y_pred_aligned, digits=3), language="text")
else:
    st.info("Upload test labels CSV to compute metrics, confusion matrix, and classification report.")
