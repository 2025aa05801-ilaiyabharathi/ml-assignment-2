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
# Paths (files are in model/ folder)
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
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14, fontweight='bold')

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
            st.image(COMPARISON_PNG, caption="Model Performance Comparison")
        else:
            st.info("model_comparison.png not found in repo.")

st.markdown("---")

if x_file is None:
    st.info("📁 Upload a test features CSV to begin.")
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

# Show selected model name prominently
st.header(f"🎯 Predictions with: {model_name}")

# If labels provided, calculate and show metrics FIRST
has_labels = y_file is not None
y_true = None
correct_count = 0
wrong_count = 0

if has_labels:
    y_true_df = pd.read_csv(y_file)
    # Accept first column as target
    y_true = y_true_df.iloc[:, 0].values

    # Safety: align lengths
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred_aligned = np.array(y_pred)[:n]
    y_proba_aligned = y_proba[:n] if y_proba is not None else None

    # Calculate correct/wrong
    correct_predictions = (y_true == y_pred_aligned)
    correct_count = correct_predictions.sum()
    wrong_count = n - correct_count

    # ===== SHOW METRICS AT THE TOP =====
    st.markdown("---")
    st.subheader("📊 Model Performance Summary")
    
    # Show correct/wrong counts
    col_summary1, col_summary2, col_summary3 = st.columns(3)
    with col_summary1:
        st.metric("✅ Correct Predictions", f"{correct_count:,}")
    with col_summary2:
        st.metric("❌ Wrong Predictions", f"{wrong_count:,}")
    with col_summary3:
        st.metric("📈 Total Predictions", f"{n:,}")

    # Compute all metrics
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

    # Show 6 metrics in a row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("AUC", f"{auc:.4f}" if auc is not None else "N/A")
    m3.metric("Precision", f"{prec:.4f}")
    m4.metric("Recall", f"{rec:.4f}")
    m5.metric("F1 Score", f"{f1:.4f}")
    m6.metric("MCC", f"{mcc:.4f}")

    st.markdown("---")

# ===== PREDICTIONS TABLE =====
st.subheader("📋 Detailed Predictions")

# Build predictions dataframe
pred_df = pd.DataFrame({
    "Prediction": y_pred.astype(int) if np.issubdtype(np.array(y_pred).dtype, np.number) else y_pred
})

if y_proba is not None:
    pred_df["Confidence"] = y_proba

if has_labels and y_true is not None:
    # Add actual and correctness columns
    n = min(len(y_true), len(y_pred))
    pred_df = pred_df.iloc[:n].copy()
    pred_df["Actual"] = y_true[:n]
    pred_df["Correct"] = (pred_df["Prediction"] == pred_df["Actual"]).map({True: "✅ Correct", False: "❌ Wrong"})
    
    # Reorder columns for better display
    if "Confidence" in pred_df.columns:
        pred_df = pred_df[["Prediction", "Actual", "Correct", "Confidence"]]
    else:
        pred_df = pred_df[["Prediction", "Actual", "Correct"]]

# Show first 100 rows (or configurable)
display_rows = st.slider("Number of rows to display", min_value=10, max_value=min(500, len(pred_df)), value=50, step=10)
st.dataframe(pred_df.head(display_rows), use_container_width=True)

# Download predictions
csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download all predictions as CSV",
    data=csv_bytes,
    file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
    mime="text/csv",
)

# ===== CONFUSION MATRIX & CLASSIFICATION REPORT (if labels provided) =====
if has_labels:
    st.markdown("---")
    st.subheader("📊 Detailed Analysis")
    
    c1, c2 = st.columns([1, 1.2])
    with c1:
        cm = confusion_matrix(y_true, y_pred_aligned)
        st.pyplot(plot_confusion(cm))
    with c2:
        st.text("Classification Report")
        st.code(classification_report(y_true, y_pred_aligned, digits=3), language="text")
else:
    st.markdown("---")
    st.info("💡 Upload test labels CSV to see accuracy, confusion matrix, and classification report.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Multi-Model Classification Dashboard | Adult Census Income Dataset</p>
    <p>Built for ML Assignment 2 | M.Tech (AIML/DSE) | BITS Pilani</p>
</div>
""", unsafe_allow_html=True)
