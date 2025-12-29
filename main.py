import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# --- CONFIGURATION ---
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

def load_and_preprocess_data():
    """Loads data, handles missing values, maps targets, and scales features."""
    print("🔄 Loading and preprocessing data...")
    df = pd.read_csv(DATA_URL, names=COLUMNS, na_values="?")
    
    # Drop missing values
    df.dropna(inplace=True)
    
    # Binary classification: 0 = Healthy, 1-4 = Disease -> mapped to 1
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Visualize Correlation (Professional Touch)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    print("✅ Correlation matrix saved as 'correlation_matrix.png'")

    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Trains a model, calculates metrics, and performs Cross-Validation."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-Validation (Scientific Robustness)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
    cv_mean = cv_scores.mean()
    
    print(f"\n🔹 {model_name} Results:")
    print(f"   Accuracy: {acc:.2f}")
    print(f"   Recall (Sensitivity): {recall:.2f}")
    print(f"   CV Mean Recall (5-Fold): {cv_mean:.2f}")
    print("-" * 30)
    
    return {
        "Accuracy": acc, 
        "Recall": recall, 
        "F1-Score": f1, 
        "CV-Recall": cv_mean,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "model_obj": model
    }

def plot_results(results, y_test):
    """Generates professional comparison plots and Confusion Matrices."""
    
    # 1. Bar Chart Comparison
    metrics_df = pd.DataFrame(results).T[["Accuracy", "Recall", "CV-Recall"]]
    metrics_df.plot(kind="bar", figsize=(10, 6), color=["#3498db", "#e74c3c", "#2ecc71"])
    plt.title("Model Performance Comparison (Includes Cross-Validation)")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.savefig("model_comparison_pro.png")
    print("✅ Performance chart saved as 'model_comparison_pro.png'")

    # 2. Confusion Matrices & ROC Curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i, (name, res) in enumerate(results.items()):
        # Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, i])
        axes[0, i].set_title(f"{name} - Confusion Matrix")
        axes[0, i].set_xlabel("Predicted")
        axes[0, i].set_ylabel("Actual")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        roc_auc = auc(fpr, tpr)
        axes[1, i].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        axes[1, i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1, i].set_title(f"{name} - ROC Curve")
        axes[1, i].set_xlabel('False Positive Rate')
        axes[1, i].set_ylabel('True Positive Rate')
        axes[1, i].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("advanced_analysis.png")
    print("✅ Advanced analysis plots saved as 'advanced_analysis.png'")

if __name__ == "__main__":
    # 1. Data Setup
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # 2. Model Definitions
    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # 3. Training & Evaluation
    results_data = {}
    for name, model in models.items():
        results_data[name] = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        
    # 4. Visualization
    plot_results(results_data, y_test)
    
    print("\n🚀 Project Execution Complete! Check the generated PNG files.")
