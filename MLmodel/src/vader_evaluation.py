"""
VADER Sentiment Analysis Evaluation & Model Comparison
Compares VADER with TF-IDF + Logistic Regression model
Generates formatted output using LLM
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from nltk.sentiment import SentimentIntensityAnalyzer

import requests

# ----------------------------
# SETUP
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config
TEXT_COLUMN = "text"
LABEL_COLUMN = "sentiment"

# Download VADER lexicon if needed
try:
    import nltk
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    import nltk
    nltk.download('vader_lexicon')

# ----------------------------
# LOAD DATA
# ----------------------------
def load_data():
    """Load cleaned CSV data"""
    try:
        # Find cleaned CSV
        csv_files = [f for f in os.listdir(DATA_DIR) if 'cleaned' in f.lower() and f.endswith('.csv')]
        if csv_files:
            data_path = os.path.join(DATA_DIR, csv_files[0])
        else:
            # Fallback to first CSV
            csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
            data_path = os.path.join(DATA_DIR, csv_files[0])
        
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path, encoding='latin-1')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

# ----------------------------
# PREPARE DATA
# ----------------------------
def prepare_data(df):
    """Prepare data for evaluation"""
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    
    # Map sentiment labels to numeric
    sentiment_map = {"positive": 1, "negative": 0, "neutral": 2}
    if LABEL_COLUMN in df.columns:
        df['sentiment_numeric'] = df[LABEL_COLUMN].map(lambda x: sentiment_map.get(str(x).lower(), -1))
    else:
        print(f"Warning: {LABEL_COLUMN} column not found. Available columns: {df.columns.tolist()}")
        return None, None, None, None
    
    # Split data (same as training)
    X = df[TEXT_COLUMN]
    y = df['sentiment_numeric']
    
    # Check class distribution
    class_counts = y.value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    
    # Only stratify if all classes have at least 2 samples
    stratify = None
    if (class_counts >= 2).all():
        stratify = y
    else:
        print("Warning: Some classes have < 2 samples. Not using stratified split.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    
    return X_test, y_test, df, sentiment_map

# ----------------------------
# VADER EVALUATION
# ----------------------------
def evaluate_vader(X_test, y_test, sentiment_map):
    """Evaluate VADER sentiment analysis"""
    sia = SentimentIntensityAnalyzer()
    
    # Map VADER scores to labels
    reverse_map = {v: k for k, v in sentiment_map.items()}
    
    vader_predictions = []
    vader_scores = []
    
    for text in X_test:
        scores = sia.polarity_scores(text)
        compound = scores['compound']
        vader_scores.append(scores)
        
        # Convert compound score to label
        if compound >= 0.05:
            label = sentiment_map['positive']
        elif compound <= -0.05:
            label = sentiment_map['negative']
        else:
            label = sentiment_map['neutral']
        
        vader_predictions.append(label)
    
    vader_predictions = np.array(vader_predictions)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, vader_predictions),
        'precision': precision_score(y_test, vader_predictions, average='weighted', zero_division=0),
        'recall': recall_score(y_test, vader_predictions, average='weighted', zero_division=0),
        'f1': f1_score(y_test, vader_predictions, average='weighted', zero_division=0)
    }
    
    return vader_predictions, vader_scores, metrics

# ----------------------------
# LOGISTIC REGRESSION EVALUATION
# ----------------------------
def evaluate_logistic(X_test, y_test):
    """Evaluate TF-IDF + Logistic Regression model"""
    try:
        tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
        model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.joblib"))
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return None, None
    
    X_test_tfidf = tfidf.transform(X_test)
    lr_predictions = model.predict(X_test_tfidf)
    
    # Convert string predictions to numeric if necessary
    sentiment_map = {"positive": 1, "negative": 0, "neutral": 2}
    if isinstance(lr_predictions[0], str):
        lr_predictions_numeric = np.array([sentiment_map.get(str(p).lower(), -1) for p in lr_predictions])
    else:
        lr_predictions_numeric = lr_predictions
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, lr_predictions_numeric),
        'precision': precision_score(y_test, lr_predictions_numeric, average='weighted', zero_division=0),
        'recall': recall_score(y_test, lr_predictions_numeric, average='weighted', zero_division=0),
        'f1': f1_score(y_test, lr_predictions_numeric, average='weighted', zero_division=0)
    }
    
    return lr_predictions_numeric, metrics

# ----------------------------
# COMPARE MODELS
# ----------------------------
def compare_models(vader_preds, lr_preds, y_test, vader_metrics, lr_metrics):
    """Compare VADER vs Logistic Regression"""
    comparison = {
        'model_vader': vader_metrics,
        'model_logistic_regression': lr_metrics,
        'agreement': (vader_preds == lr_preds).sum() / len(y_test) * 100,
        'test_size': len(y_test)
    }
    
    print("\n" + "="*60)
    print("MODEL COMPARISON: VADER vs TF-IDF + Logistic Regression")
    print("="*60)
    
    print("\nVADER Metrics:")
    for key, val in vader_metrics.items():
        print(f"  {key.upper()}: {val:.4f}")
    
    print("\nLogistic Regression Metrics:")
    for key, val in lr_metrics.items():
        print(f"  {key.upper()}: {val:.4f}")
    
    print(f"\nModel Agreement: {comparison['agreement']:.2f}%")
    
    return comparison

# ----------------------------
# LLM FORMULATION
# ----------------------------
def formulate_with_llm(comparison, vader_metrics, lr_metrics, output_file):
    """Use LLM to generate formatted insights"""
    
    prompt = f"""
Based on the following model comparison results, generate a comprehensive analysis report:

VADER Sentiment Analysis Performance:
- Accuracy: {vader_metrics['accuracy']:.4f}
- Precision: {vader_metrics['precision']:.4f}
- Recall: {vader_metrics['recall']:.4f}
- F1 Score: {vader_metrics['f1']:.4f}

TF-IDF + Logistic Regression Performance:
- Accuracy: {lr_metrics['accuracy']:.4f}
- Precision: {lr_metrics['precision']:.4f}
- Recall: {lr_metrics['recall']:.4f}
- F1 Score: {lr_metrics['f1']:.4f}

Model Agreement Rate: {comparison['agreement']:.2f}%

Please provide:
1. Key findings and model performance comparison
2. Strengths and weaknesses of each approach
3. Recommendations for which model to use in production
4. Suggestions for improvement

Format as a clear, professional report.
"""

    # Try OpenRouter first, fallback to Gemini
    llm_response = call_llm(prompt)
    
    if llm_response:
        return llm_response
    else:
        return generate_default_report(comparison, vader_metrics, lr_metrics)

def call_llm(prompt):
    """Call LLM via OpenRouter with Gemini fallback"""
    
    # Try OpenRouter
    openrouter_keys = [
        os.getenv('OPENROUTER_KEY_1'),
        os.getenv('OPENROUTER_KEY_2')
    ]
    
    for key in openrouter_keys:
        if not key:
            continue
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Project BlackFlag"
                },
                json={
                    "model": "openai/gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"OpenRouter error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"OpenRouter attempt failed: {e}")
    
    # Try Gemini
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini error: {e}")
    
    return None

def generate_default_report(comparison, vader_metrics, lr_metrics):
    """Generate report if LLM fails"""
    report = f"""
SENTIMENT ANALYSIS MODEL COMPARISON REPORT
{'='*60}

EXECUTIVE SUMMARY
{'-'*60}
This analysis compares two sentiment analysis approaches:
1. VADER (Valence Aware Dictionary and sEntiment Reasoner)
2. TF-IDF + Logistic Regression

PERFORMANCE METRICS
{'-'*60}

VADER Sentiment Analysis:
  Accuracy:  {vader_metrics['accuracy']:.4f} ({vader_metrics['accuracy']*100:.2f}%)
  Precision: {vader_metrics['precision']:.4f}
  Recall:    {vader_metrics['recall']:.4f}
  F1 Score:  {vader_metrics['f1']:.4f}

TF-IDF + Logistic Regression:
  Accuracy:  {lr_metrics['accuracy']:.4f} ({lr_metrics['accuracy']*100:.2f}%)
  Precision: {lr_metrics['precision']:.4f}
  Recall:    {lr_metrics['recall']:.4f}
  F1 Score:  {lr_metrics['f1']:.4f}

MODEL COMPARISON
{'-'*60}
Agreement Rate: {comparison['agreement']:.2f}%
Test Set Size: {comparison['test_size']} samples

WINNER: {'TF-IDF + Logistic Regression' if lr_metrics['accuracy'] > vader_metrics['accuracy'] else 'VADER'}
Accuracy Difference: {abs(lr_metrics['accuracy'] - vader_metrics['accuracy']):.4f}

KEY INSIGHTS
{'-'*60}

1. VADER Strengths:
   - Rule-based lexicon approach
   - Fast and lightweight
   - No training required
   - Good for real-time sentiment analysis

2. VADER Limitations:
   - May miss context-specific sentiment
   - Limited to predefined vocabulary
   - Accuracy: {vader_metrics['accuracy']:.4f}

3. Logistic Regression Strengths:
   - Learns from training data
   - Captures domain-specific patterns
   - Higher accuracy: {lr_metrics['accuracy']:.4f}
   - Good generalization

4. Logistic Regression Limitations:
   - Requires labeled training data
   - More computational overhead
   - Training time

RECOMMENDATIONS
{'-'*60}

1. Production Deployment:
   Use {'TF-IDF + Logistic Regression' if lr_metrics['accuracy'] > vader_metrics['accuracy'] else 'VADER'} for production based on accuracy.

2. Implementation Strategy:
   - Primary: {'TF-IDF + Logistic Regression' if lr_metrics['accuracy'] > vader_metrics['accuracy'] else 'VADER'}
   - Fallback: {'VADER' if lr_metrics['accuracy'] > vader_metrics['accuracy'] else 'TF-IDF + Logistic Regression'}

3. Improvement Opportunities:
   - Ensemble method combining both approaches
   - Hyperparameter tuning for better performance
   - Extended training data collection
   - Regular model retraining with new data

CONCLUSION
{'-'*60}
Both models have merit. The {'TF-IDF + Logistic Regression' if lr_metrics['accuracy'] > vader_metrics['accuracy'] else 'VADER'} approach 
shows superior performance for this dataset and is recommended for production use.
"""
    return report

# ----------------------------
# SAVE RESULTS
# ----------------------------
def save_results(comparison, vader_metrics, lr_metrics, X_test, y_test, vader_preds, lr_preds, llm_report):
    """Save evaluation results to CSV and JSON"""
    
    # Detailed results DataFrame
    results_df = pd.DataFrame({
        'text': X_test.values,
        'actual_label': y_test.values,
        'vader_prediction': vader_preds,
        'lr_prediction': lr_preds,
        'models_agree': vader_preds == lr_preds
    })
    
    # Save detailed results
    results_csv = os.path.join(OUTPUT_DIR, 'model_comparison_detailed.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\nDetailed results saved to: {results_csv}")
    
    # Save metrics summary
    metrics_summary = {
        'vader': vader_metrics,
        'logistic_regression': lr_metrics,
        'comparison': {
            'agreement_percent': comparison['agreement'],
            'test_size': comparison['test_size']
        }
    }
    
    metrics_json = os.path.join(OUTPUT_DIR, 'model_comparison_metrics.json')
    with open(metrics_json, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Metrics saved to: {metrics_json}")
    
    # Save LLM report
    report_file = os.path.join(OUTPUT_DIR, 'model_comparison_report.txt')
    with open(report_file, 'w') as f:
        f.write(llm_report)
    print(f"LLM Report saved to: {report_file}")
    
    # Save summary CSV
    summary_df = pd.DataFrame([
        {
            'Model': 'VADER',
            'Accuracy': vader_metrics['accuracy'],
            'Precision': vader_metrics['precision'],
            'Recall': vader_metrics['recall'],
            'F1_Score': vader_metrics['f1']
        },
        {
            'Model': 'TF-IDF + Logistic Regression',
            'Accuracy': lr_metrics['accuracy'],
            'Precision': lr_metrics['precision'],
            'Recall': lr_metrics['recall'],
            'F1_Score': lr_metrics['f1']
        }
    ])
    
    summary_csv = os.path.join(OUTPUT_DIR, 'model_comparison_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved to: {summary_csv}")

# ----------------------------
# MAIN EXECUTION
# ----------------------------
def main():
    print("="*60)
    print("VADER Sentiment Analysis & Model Comparison")
    print("="*60)
    
    # Load and prepare data
    print("\n[1/5] Loading data...")
    df = load_data()
    X_test, y_test, df, sentiment_map = prepare_data(df)
    
    if X_test is None:
        print("Failed to prepare data")
        sys.exit(1)
    
    # Evaluate VADER
    print("\n[2/5] Evaluating VADER...")
    vader_preds, vader_scores, vader_metrics = evaluate_vader(X_test, y_test, sentiment_map)
    
    # Evaluate Logistic Regression
    print("\n[3/5] Evaluating Logistic Regression...")
    lr_preds, lr_metrics = evaluate_logistic(X_test, y_test)
    
    if lr_preds is None:
        print("Failed to evaluate logistic regression model")
        sys.exit(1)
    
    # Compare models
    print("\n[4/5] Comparing models...")
    comparison = compare_models(vader_preds, lr_preds, y_test, vader_metrics, lr_metrics)
    
    # LLM formulation
    print("\n[5/5] Generating LLM report...")
    llm_report = formulate_with_llm(comparison, vader_metrics, lr_metrics, OUTPUT_DIR)
    
    # Save results
    print("\nSaving results...")
    save_results(comparison, vader_metrics, lr_metrics, X_test, y_test, vader_preds, lr_preds, llm_report)
    
    # Print report
    print("\n" + "="*60)
    print("LLM GENERATED REPORT")
    print("="*60)
    print(llm_report)
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
