import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

# =======================
# CONFIGURATION
# =======================
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

RANDOM_STATE = 42

# =======================
# DATA LOADING
# =======================
def load_data():
    """
    Loads dataset, handles missing values,
    and prepares features and target variable.
    """
    print(" Loading dataset...")
    df = pd.read_csv(DATA_URL, names=COLUMNS, na_values="?")
    df.dropna(inplace=True)

    # Binary classification: 0 = Healthy, 1 = Disease
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    # Correlation Matrix (for analysis & report)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")

    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

# =======================
# MODEL EVALUATION
# =======================
def evaluate_model(pipeline, X_train, X_test, y_train, y_test, model_name):
    """
    Trains the pipeline, evaluates performance,
    and performs cross-validation using Recall.
    """
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Cross-validation WITHOUT data leakage
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="recall"
    )

    print(f"\n🔹 {model_name}")
    print(f"   Accuracy : {acc:.2f}")
    print(f"   Recall   : {recall:.2f}")
    print(f"   CV Recall: {cv_scores.mean():.2f}")
    print("-" * 35)

    return {
        "Accuracy": acc,
        "Recall": recall,
        "CV-Recall": cv_scores.mean(),
        "y_pred": y_pred,
        "y_prob": y_prob
    }

# =======================
# VISUALIZATION
# =======================
def plot_results(results, y_test):
    """
    Plots performance comparison,
    confusion matrices, and ROC curves.
    """
    # --- Bar Chart ---
    metrics_df = pd.DataFrame(results).T[
        ["Accuracy", "Recall", "CV-Recall"]
    ]

    metrics_df.plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0, 1),
        grid=True
    )

    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("model_comparison.png")

    # --- Confusion Matrix & ROC ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, i])
        axes[0, i].set_title(f"{name} - Confusion Matrix")

        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        roc_auc = auc(fpr, tpr)

        axes[1, i].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        axes[1, i].plot([0, 1], [0, 1], linestyle="--")
        axes[1, i].set_title(f"{name} - ROC Curve")
        axes[1, i].legend()

    plt.tight_layout()
    plt.savefig("advanced_analysis.png")

# =======================
# MAIN
# =======================
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_data()

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),

        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),

        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE
            ))
        ])
    }

    results = {}

    for name, pipeline in models.items():
        results[name] = evaluate_model(
            pipeline, X_train, X_test, y_train, y_test, name
        )

    plot_results(results, y_test)

    print("\n Analysis complete. Files saved successfully.")
