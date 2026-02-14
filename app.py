import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="ML Model Classifier",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Multi-Model Classification Dashboard")
st.markdown("""
This interactive app demonstrates 6 different classification models trained on the Adult Census Income dataset.
Upload the pre-processed test data and select a model to see predictions and evaluation metrics.
""")

# Sidebar for model selection
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Model selection
model_options = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
#    "K-Nearest Neighbors": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys())
)

# Load preprocessing pipeline
@st.cache_resource
def load_preprocessing():
    try:
        scaler = joblib.load('model/preprocessing_pipeline.pkl')
        return scaler
    except FileNotFoundError:
        st.error("Preprocessing pipeline not found! Please run model_training.ipynb first.")
        return None

# Load selected model
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please run model_training.ipynb first.")
        return None

# Load model comparison data
@st.cache_data
def load_comparison_data():
    try:
        comparison_df = pd.read_csv('model/model_comparison.csv')
        return comparison_df
    except FileNotFoundError:
        return None

# Main content
scaler = load_preprocessing()
model = load_model(model_options[selected_model_name])

# File uploader
st.sidebar.markdown("---")
st.sidebar.header("📁 Upload Test Data")
st.sidebar.info("""
⚠️ **Important:** Please upload the `test_data.csv` file from the `model/` folder.

This file contains pre-processed data with:
- Numerical features only
- No missing values
- Same format as training data
""")

uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv",
    type=['csv'],
    help="Upload the test_data.csv file generated from model_training.ipynb"
)

# Show model comparison
st.header("📊 Model Performance Comparison")
comparison_df = load_comparison_data()

if comparison_df is not None:
    st.dataframe(
        comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']),
        use_container_width=True
    )
else:
    st.info("Model comparison data not available. Run model_training.ipynb first.")

st.markdown("---")

# Prediction section
st.header(f"🎯 Predictions with {selected_model_name}")

if uploaded_file is not None and model is not None and scaler is not None:
    try:
        # Load uploaded data
        data = pd.read_csv(uploaded_file)
        st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
        
        # Show sample data
        with st.expander("👀 View uploaded data sample"):
            st.dataframe(data.head())
        
        # Check if we have labels file
        try:
            y_test = pd.read_csv('model/test_labels.csv')
            has_target = True
        except FileNotFoundError:
            y_test = None
            has_target = False
            st.warning("⚠️ Test labels file not found. Predictions will be made without evaluation metrics.")
        
        X_test = data
        
        # Preprocess data
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        if st.button("🚀 Run Predictions", type="primary"):
            with st.spinner("Making predictions..."):
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                # Create results dataframe
                results_df = X_test.copy()
                results_df['Predicted_Class'] = y_pred
                results_df['Prediction_Confidence'] = y_pred_proba.max(axis=1)
                
                if has_target:
                    results_df['Actual_Class'] = y_test.values
                    results_df['Correct'] = (y_pred == y_test.values.flatten())
                
                # Display results
                st.subheader("📋 Prediction Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Download predictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
                # Evaluation metrics (only if target is available)
                if has_target:
                    st.markdown("---")
                    st.subheader("📈 Evaluation Metrics")
                    
                    # Calculate metrics
                    y_test_flat = y_test.values.flatten()
                    accuracy = accuracy_score(y_test_flat, y_pred)
                    
                    # Handle AUC for binary/multiclass
                    n_classes = len(np.unique(y_test_flat))
                    if n_classes == 2:
                        auc = roc_auc_score(y_test_flat, y_pred_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_test_flat, y_pred_proba, multi_class='ovr')
                    
                    avg_method = 'binary' if n_classes == 2 else 'weighted'
                    precision = precision_score(y_test_flat, y_pred, average=avg_method, zero_division=0)
                    recall = recall_score(y_test_flat, y_pred, average=avg_method, zero_division=0)
                    f1 = f1_score(y_test_flat, y_pred, average=avg_method, zero_division=0)
                    mcc = matthews_corrcoef(y_test_flat, y_pred)
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    col4, col5, col6 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    with col2:
                        st.metric("AUC Score", f"{auc:.4f}")
                    with col3:
                        st.metric("Precision", f"{precision:.4f}")
                    with col4:
                        st.metric("Recall", f"{recall:.4f}")
                    with col5:
                        st.metric("F1 Score", f"{f1:.4f}")
                    with col6:
                        st.metric("MCC Score", f"{mcc:.4f}")
                    
                    # Confusion Matrix
                    st.markdown("---")
                    st.subheader("🔢 Confusion Matrix")
                    
                    cm = confusion_matrix(y_test_flat, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted Label')
                    ax.set_ylabel('True Label')
                    ax.set_title(f'Confusion Matrix - {selected_model_name}')
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.markdown("---")
                    st.subheader("📊 Classification Report")
                    
                    report = classification_report(y_test_flat, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Make sure you uploaded the correct test_data.csv file from the model/ folder.")

else:
    st.info("👆 Please upload the test_data.csv file to make predictions.")
    
    # Show expected format
    st.subheader("📝 File Requirements")
    st.markdown("""
    **Required file:** `test_data.csv` from the `model/` folder
    
    This file contains:
    - 14 numerical features (age, workclass, fnlwgt, education, etc.)
    - Pre-processed and encoded categorical variables
    - No missing values
    - No target column (labels are in test_labels.csv)
    
    **To get this file:**
    1. Run the `model_training.ipynb` notebook
    2. Find `test_data.csv` in the `model/` directory
    3. Upload it here
    """)
    
    # Show sample structure
    st.markdown("**Expected column structure:**")
    sample_cols = {
        'age': [38, 50, 38],
        'workclass': [4, 3, 2],
        'fnlwgt': [215646, 234721, 162208],
        'education': [11, 12, 9],
        '...': ['...', '...', '...'],
        'native-country': [38, 38, 38]
    }
    st.dataframe(pd.DataFrame(sample_cols))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Multi-Model Classification Dashboard | Adult Census Income Dataset</p>
    <p>Built for ML Assignment 2 | M.Tech (AIML/DSE) | BITS Pilani</p>
</div>
""", unsafe_allow_html=True)
