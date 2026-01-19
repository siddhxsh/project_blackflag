"""
VADER Sentiment Analysis Baseline
Compares with the ML model (TF-IDF + Logistic Regression)
Uses the same dataset and train/test split for fair comparison
"""

import os
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Download VADER lexicon (first time only)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download("vader_lexicon")

# ----------------------------
# CONFIG
# ----------------------------
TEXT_COLUMN = "text"  # Text column in cleaned CSV
LABEL_COLUMN = "sentiment"
FILE_NAME = "amazon_cleaned.csv"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", FILE_NAME)

# ----------------------------
# LOAD DATA
# ----------------------------
def load_and_prepare_data():
    """Load and prepare data for VADER analysis"""
    df = pd.read_csv(DATA_PATH)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    
    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]
    
    # Same train-test split as ML model (random_state=42, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    return X_test, y_test

# ----------------------------
# VADER SENTIMENT ANALYZER
# ----------------------------
def get_vader_predictions(X_test):
    """Get VADER sentiment predictions"""
    sia = SentimentIntensityAnalyzer()
    
    def vader_predict(text):
        score = sia.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    return X_test.apply(vader_predict)

# ----------------------------
# EVALUATION METRICS
# ----------------------------
def evaluate_vader(y_test, y_pred):
    """Calculate VADER evaluation metrics"""
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

if __name__ == "__main__":
    X_test, y_test = load_and_prepare_data()
    y_pred = get_vader_predictions(X_test)
    metrics = evaluate_vader(y_test, y_pred)
    
    print("\n" + "="*50)
    print("VADER BASELINE RESULTS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
