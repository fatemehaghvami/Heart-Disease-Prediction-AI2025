# Heart Disease Detection Using Machine Learning

## Project Description
This project detects the presence of heart disease using supervised machine learning models.
Three classifiers are implemented and compared with a focus on **Recall (Sensitivity)**,
which is a critical metric in medical diagnosis problems.

The project is developed for academic purposes.

---

## Dataset
- **Source:** UCI Machine Learning Repository
- **Dataset:** Cleveland Heart Disease Dataset
- **Number of Features:** 13
- **Target Variable:**
  - `0` → No heart disease
  - `1` → Presence of heart disease

Missing values (`?`) are removed during preprocessing.

---

## Data Preprocessing
The following preprocessing steps are applied:
- Loading data from the UCI repository
- Removing rows with missing values
- Mapping multi-class targets into binary labels
- Feature scaling using **StandardScaler**
- Splitting data into training and testing sets (80% / 20%)

A feature correlation matrix is generated and saved for exploratory analysis.

---

## Machine Learning Models
The following models are trained and evaluated:

- **Logistic Regression**
- **K-Nearest Neighbors (k = 5)**
- **Random Forest (100 trees)**

Each model is evaluated using the same training and test sets.

---

## Evaluation Metrics
Model performance is evaluated using:
- Accuracy
- Recall (Sensitivity)
- F1-Score
- 5-Fold Cross-Validation Recall

The following visual outputs are generated:
- Feature correlation heatmap
- Model performance comparison bar chart
- Confusion matrices
- ROC curves with AUC values

---

## Project Structure
