# ML Assignment 2 - Multi-Model Classification

## Problem Statement

This project implements and compares 6 different machine learning classification models on the **Adult Census Income dataset**. The goal is to predict whether an individual's annual income exceeds $50K based on demographic and employment features. The dataset contains 14 features and 30,162 instances, addressing a binary classification problem in the socioeconomic domain.

---

## Dataset Description

**Dataset Name**: Adult Census Income Dataset  
**Source**: UCI Machine Learning Repository  
**Link**: https://archive.ics.uci.edu/ml/datasets/adult

### Dataset Characteristics:
- **Total Instances**: 30,162
- **Total Features**: 14
- **Target Variable**: Income level (0 = ≤50K, 1 = >50K)
- **Problem Type**: Binary Classification
- **Class Distribution**: 
  - Class 0 (≤50K): 22,654 instances (75.1%)
  - Class 1 (>50K): 7,508 instances (24.9%)

### Feature Overview:
| Feature Name | Type | Description |
|-------------|------|-------------|
| age | Numerical | Age in years |
| workclass | Categorical | Employment sector (encoded) |
| fnlwgt | Numerical | Final sampling weight |
| education | Categorical | Highest education level (encoded) |
| education-num | Numerical | Education level as number |
| marital-status | Categorical | Marital status (encoded) |
| occupation | Categorical | Job type (encoded) |
| relationship | Categorical | Family relationship (encoded) |
| race | Categorical | Race (encoded) |
| sex | Categorical | Gender (encoded) |
| capital-gain | Numerical | Capital gains |
| capital-loss | Numerical | Capital losses |
| hours-per-week | Numerical | Working hours per week |
| native-country | Categorical | Country of origin (encoded) |

### Data Preprocessing:
- Missing value treatment: Rows with missing values removed
- Categorical encoding: Label encoding applied to all categorical features
- Feature scaling: StandardScaler applied to normalize features
- Train-Test Split: 80-20 split with stratification

---

## Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8175 | 0.8501 | 0.7135 | 0.4461 | 0.5490 | 0.4612 |
| Decision Tree | 0.8510 | 0.8848 | 0.7494 | 0.6032 | 0.6683 | 0.5795 |
| kNN | 0.8190 | 0.8498 | 0.6530 | 0.5826 | 0.6158 | 0.4993 |
| Naive Bayes | 0.7978 | 0.8498 | 0.6986 | 0.3302 | 0.4486 | 0.3798 |
| Random Forest (Ensemble) | 0.8525 | 0.9136 | 0.7977 | 0.5459 | 0.6482 | 0.5751 |
| XGBoost (Ensemble) | 0.8551 | 0.9140 | 0.7404 | 0.6438 | 0.6887 | 0.5974 |

---

## Model Performance Observations

### Individual Model Analysis:

**1. Logistic Regression:**
Good starting point with 81.8% accuracy, but struggles to find high earners (only 45% recall). It's playing it safe by mostly predicting the common class (≤50K income). Works fast and simple, but misses too many high-income people to be fully useful.

**2. Decision Tree:**
Strong performer at 85.1% accuracy. Does a much better job finding high earners (60% recall) than Logistic Regression. When it predicts someone earns >50K, it's right about 75% of the time. Good balance between precision and recall.

**3. K-Nearest Neighbors (kNN):**
Middle performer at 81.9% accuracy. Finds a decent number of high earners (58% recall) but precision is lower (65%). With 14 features and 30,000 people, it struggles a bit. Also really slow to make predictions.

**4. Naive Bayes:**
Weakest performer at 79.8% accuracy. Worst at finding high earners (only 33% recall). Too cautious and misses most actual high-income individuals. Super fast to train though, which makes it good for quick baseline tests.

**5. Random Forest (Ensemble):**
Most precise model (80%) with great accuracy (85.3%) and excellent AUC (0.914). When it says someone earns >50K, it's usually right. But it's cautious and misses some actual high earners (55% recall). Good if you hate false alarms.

**6. XGBoost (Ensemble):**
**Winner overall.** Best at finding high earners (64% recall) while keeping excellent accuracy (85.5%) and AUC (0.914). Most balanced across all measures with best F1 (0.689) and MCC (0.597). Handles the imbalanced data better than everything else.

### Key Takeaways:

**Best Model:** XGBoost wins. It finds the most high-income people without sacrificing accuracy. Best overall balance with highest F1 and MCC scores.

**Ensemble Power:** Random Forest and XGBoost significantly outperformed basic models, especially in AUC scores (0.91+ vs 0.85). Combining multiple models clearly works better.

**Biggest Difference:** Recall varied dramatically (33% to 64%). Shows how differently each model handles imbalanced data.

**Trade-off:** Random Forest is super precise (80%) but cautious (55% recall). XGBoost is slightly less precise (74%) but finds way more people (64% recall). For this task, finding high earners matters more, so XGBoost wins.

**Surprise:** Naive Bayes performed worst despite being popular. Too conservative with predictions. kNN also struggled with the high-dimensional space.

**The Imbalance Problem:** All models are decent at accuracy (80-85%) but varied wildly at finding high earners (33-64% recall). That's because 75% of people earn ≤50K, so models can get high accuracy by just guessing "low income" often.

**Bottom Line:** Use XGBoost. It's the most reliable at actually identifying both income groups with the best overall metrics.

---

## Live Deployment

### Streamlit App
**Live Link**: https://ml-assignment-2-ilaiya.streamlit.app/

### GitHub Repository
**Repository Link**: https://github.com/2025aa05801-ilaiyabharathi/ml-assignment-2

### Features:
- Interactive model selection dropdown
- CSV file upload for test data
- Real-time predictions with probability scores
- Comprehensive evaluation metrics display
- Confusion matrix visualization
- Classification report
- Downloadable prediction results

---

## Repository Structure

```
ml-assignment-2/
│
├── app.py                          # Streamlit web application
├── model_training_ML.ipynb         # Jupyter notebook with model training
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
└── model/
    ├── logistic.pkl                # Trained Logistic Regression model
    ├── decision_tree.pkl           # Trained Decision Tree model
    ├── knn.pkl                     # Trained KNN model
    ├── naive_bayes.pkl             # Trained Naive Bayes model
    ├── random_forest.pkl           # Trained Random Forest model
    ├── xgboost.pkl                 # Trained XGBoost model
    ├── preprocessing_pipeline.pkl  # Scaler for data preprocessing
    ├── model_comparison.csv        # Metrics comparison table
    ├── model_comparison.png        # Performance visualization
    ├── test_data.csv               # Sample test data
    └── test_labels.csv             # Test labels for evaluation
```

---

## Evaluation Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **AUC (Area Under ROC Curve)**: Model's ability to distinguish between classes 
- **Precision**: When model predicts high income, how often is it correct?
- **Recall**: Of all actual high earners, how many did the model find?
- **F1 Score**: Harmonic mean of precision and recall (balances both)
- **MCC (Matthews Correlation Coefficient)**: Balanced measure considering all confusion matrix elements

---

## Author

**Name**: G Ilaiya Bharathi

**Email**: 2025aa05801@wilp.bits-pilani.ac.in

**Student ID**: 2025aa05801


