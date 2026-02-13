# Heart Disease Detection using Machine Learning (Pipeline-Based)

## 📌 Project Overview
This project implements a **heart disease detection system** using classical machine learning models.
The focus is on building a **scientifically correct and leakage-free pipeline**, suitable for **academic submission** and introductory medical decision support discussions.

A strong emphasis is placed on **Recall (Sensitivity)**, which is critical in healthcare scenarios where failing to detect a sick patient (False Negative) can be dangerous.

---

## 🧠 Models Implemented
The following models are trained and evaluated using **scikit-learn Pipelines**:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN, k=5)** – baseline model
- **Random Forest Classifier**

Each model is wrapped in a Pipeline with:
- `StandardScaler`
- The corresponding classifier

✅ This ensures **no data leakage** during training and cross-validation.

---

## 📊 Dataset
- **Source:** UCI Machine Learning Repository  
  Cleveland Heart Disease Dataset  
- **Target Variable:**
  - `0` → Healthy
  - `1` → Heart Disease  
    (Original labels 1–4 merged into a single disease class)

Missing values (`?`) are removed before training.

---

## ⚙️ Data Preprocessing
- Handling missing values (`dropna`)
- Binary target mapping (healthy vs disease)
- Feature scaling using `StandardScaler` (inside Pipeline)
- Train/Test split: **80% / 20%**
- Random State: `42`

A **feature correlation matrix** is generated and saved for analysis.

---

## 📈 Evaluation Metrics
Each model is evaluated using:

- **Accuracy**
- **Recall (Sensitivity)** ✅ *Primary Metric*
- **F1-Score**
- **5-Fold Cross-Validation Recall**

### Why Recall?
In medical diagnosis:
> False Negatives (missing a sick patient) are more critical than False Positives.

---

## 📉 Generated Outputs
Running `main.py` produces the following files:

- `correlation_matrix.png`  
  Heatmap of feature correlations
- `model_comparison.png`  
  Bar chart comparing Accuracy, Recall, and CV Recall
- `advanced_analysis.png`  
  - Confusion Matrices  
  - ROC Curves  
  - AUC values for all models

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
