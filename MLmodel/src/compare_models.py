"""
Compare VADER Baseline vs ML Model (TF-IDF + Logistic Regression)
Evaluates both models on the same test set and generates comparison metrics
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from vader_baseline import load_and_prepare_data, get_vader_predictions, evaluate_vader

# ----------------------------
# CONFIG
# ----------------------------
TEXT_COLUMN = "text"
LABEL_COLUMN = "sentiment"
FILE_NAME = "amazon_cleaned.csv"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", FILE_NAME)
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

# Check class distribution
print("Class distribution:")
print(y.value_counts())

# Filter to only classes with at least 2 samples for stratification
min_count = y.value_counts().min()
if min_count < 2:
    print(f"Warning: Minimum class count is {min_count}. Removing classes with only 1 sample.")
    # Keep only classes with at least 2 samples
    valid_classes = y.value_counts()[y.value_counts() >= 2].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]
    print(f"After filtering: {len(X)} samples remaining")
    print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if len(y.unique()) > 1 and y.value_counts().min() >= 2 else None
)

# ==========================================================
# 1️⃣ VADER BASELINE
# ==========================================================
y_pred_vader = get_vader_predictions(X_test)
vader_metrics = evaluate_vader(y_test, y_pred_vader)

# ==========================================================
# 2️⃣ ML MODEL (TF-IDF + Logistic Regression)
# ==========================================================
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.joblib"))

X_test_tfidf = tfidf.transform(X_test)
y_pred_ml = model.predict(X_test_tfidf)

ml_accuracy = accuracy_score(y_test, y_pred_ml)
ml_precision, ml_recall, ml_f1, _ = precision_recall_fscore_support(
    y_test, y_pred_ml, average="weighted"
)

ml_metrics = {
    "accuracy": float(ml_accuracy),
    "precision": float(ml_precision),
    "recall": float(ml_recall),
    "f1": float(ml_f1),
    "classification_report": classification_report(y_test, y_pred_ml)
}

# ==========================================================
# 3️⃣ COMPARISON TABLE
# ==========================================================
comparison_df = pd.DataFrame({
    "Model": ["VADER (Baseline)", "TF-IDF + Logistic Regression"],
    "Accuracy": [vader_metrics["accuracy"], ml_metrics["accuracy"]],
    "Precision": [vader_metrics["precision"], ml_metrics["precision"]],
    "Recall": [vader_metrics["recall"], ml_metrics["recall"]],
    "F1-Score": [vader_metrics["f1"], ml_metrics["f1"]]
})

# Save comparison to CSV
output_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
comparison_df.to_csv(output_path, index=False)

# ==========================================================
# 4️⃣ PRINT RESULTS
# ==========================================================
def print_comparison():
    """Print formatted comparison table"""
    print("\n" + "="*70)
    print("MODEL COMPARISON: VADER vs TF-IDF + Logistic Regression")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70)
    
    # Calculate improvement
    ml_vs_vader_acc = ((ml_metrics["accuracy"] - vader_metrics["accuracy"]) / vader_metrics["accuracy"]) * 100 if vader_metrics["accuracy"] > 0 else 0
    print(f"\nML Model Accuracy Improvement: {ml_vs_vader_acc:+.2f}%")
    print(f"Comparison saved to: {output_path}")

def get_comparison_dict():
    """Return comparison as dictionary for API response"""
    return {
        "comparison_table": comparison_df.to_dict(orient="records"),
        "vader_metrics": vader_metrics,
        "ml_metrics": ml_metrics,
        "improvement": {
            "accuracy_percent": float(((ml_metrics["accuracy"] - vader_metrics["accuracy"]) / vader_metrics["accuracy"]) * 100 if vader_metrics["accuracy"] > 0 else 0),
            "better_model": "ML Model" if ml_metrics["accuracy"] > vader_metrics["accuracy"] else "VADER"
        }
    }

if __name__ == "__main__":
    print_comparison()
