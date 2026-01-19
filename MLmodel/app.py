from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import pandas as pd
import re
import joblib
from io import StringIO
import sys
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.sentiment import SentimentIntensityAnalyzer
import requests

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from column_analyzer import analyze_columns_with_llm

# Download VADER lexicon if needed
try:
    import nltk
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    import nltk
    nltk.download('vader_lexicon')

app = Flask(__name__)
CORS(app)  # Enable CORS for Vercel frontend

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'outputs')

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def clean_data_pipeline(df, column_mapping):
    """Clean the data based on column mapping"""
    # Reverse mapping
    col_map = {v: k for k, v in column_mapping.items() if v is not None}
    
    # Rename columns
    df = df.rename(columns=col_map)
    
    # Keep only required columns
    required_cols = ["ProductName", "Price", "Rate", "Review", "Summary"]
    existing_cols = [col for col in required_cols if col in df.columns]
    df = df[existing_cols].copy()
    
    # Fill missing columns
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    
    # Normalize text
    for col in ["Review", "Summary"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    
    # Drop empty rows
    empty_mask = (df["Review"] == "") & (df["Summary"] == "")
    df = df.loc[~empty_mask].copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Validate Rate
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df = df[df["Rate"].isin([1, 2, 3, 4, 5])]
    df["Rate"] = df["Rate"].astype(int)
    
    # Clean Price
    if "Price" in df.columns:
        df["Price"] = df["Price"].astype(str).str.replace(r'[^\d]', '', regex=True)
        df["Price"] = pd.to_numeric(df["Price"], errors='coerce')
    
    # Create text column
    df["text"] = df["Summary"] + " " + df["Review"]
    
    # Clean text
    def clean_text(text):
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[\n\t\r]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    for col in ["Summary", "Review", "text"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    
    # Map sentiment
    def map_sentiment(r):
        if r >= 4: return "Positive"
        if r == 3: return "Neutral"
        return "Negative"
    
    df["sentiment"] = df["Rate"].apply(map_sentiment)
    
    return df


def generate_ml_predictions(df):
    """Generate ML predictions using trained model"""
    model_path = os.path.join(MODELS_DIR, 'logistic_model.joblib')
    vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        X = vectorizer.transform(df['text'])
        df['predicted_sentiment'] = model.predict(X)
    else:
        # If no model, use rule-based sentiment
        df['predicted_sentiment'] = df['sentiment']
    
    return df


def extract_keywords_from_df(df):
    """Extract positive and negative keywords"""
    from collections import Counter
    import re
    
    positive_texts = df[df['sentiment'] == 'Positive']['text'].tolist()
    negative_texts = df[df['sentiment'] == 'Negative']['text'].tolist()
    
    def get_words(texts):
        words = []
        for text in texts:
            words.extend(re.findall(r'\b\w+\b', text.lower()))
        return Counter(words)
    
    pos_counter = get_words(positive_texts)
    neg_counter = get_words(negative_texts)
    
    # Filter common words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'this', 'that', 'it', 'as', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    pos_counter = {k: v for k, v in pos_counter.items() if k not in stopwords and len(k) > 2}
    neg_counter = {k: v for k, v in neg_counter.items() if k not in stopwords and len(k) > 2}
    
    pos_df = pd.DataFrame([{'keyword': k, 'frequency': v} for k, v in sorted(pos_counter.items(), key=lambda x: x[1], reverse=True)[:50]])
    neg_df = pd.DataFrame([{'keyword': k, 'frequency': v} for k, v in sorted(neg_counter.items(), key=lambda x: x[1], reverse=True)[:50]])
    
    return pos_df, neg_df


def analyze_aspects(df):
    """Analyze sentiment by aspect"""
    aspects = {
        'Price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'money', 'worth'],
        'Quality': ['quality', 'durable', 'sturdy', 'build', 'material', 'construction', 'made', 'last'],
        'Delivery': ['delivery', 'shipping', 'arrived', 'package', 'received', 'time', 'fast', 'slow'],
        'Performance': ['work', 'works', 'performance', 'function', 'speed', 'efficient', 'effective']
    }
    
    results = []
    for aspect, keywords in aspects.items():
        aspect_df = df[df['text'].str.contains('|'.join(keywords), case=False, na=False)]
        if len(aspect_df) > 0:
            sentiment_counts = aspect_df['sentiment'].value_counts()
            total = len(aspect_df)
            results.append({
                'aspect': aspect,
                'positive': sentiment_counts.get('Positive', 0),
                'negative': sentiment_counts.get('Negative', 0),
                'neutral': sentiment_counts.get('Neutral', 0),
                'total_mentions': total
            })
    
    return pd.DataFrame(results)


def analyze_component_failures(df):
    """Extract component failures from negative reviews"""
    negative_df = df[df['sentiment'] == 'Negative']
    
    components = ['cable', 'connector', 'wire', 'plug', 'adapter', 'charger', 'port', 'cord', 'usb']
    failure_words = ['broke', 'broken', 'failed', 'stop', 'stopped', 'not work', 'doesnt work', 'issue', 'problem']
    
    results = []
    for component in components:
        comp_df = negative_df[negative_df['text'].str.contains(component, case=False, na=False)]
        failure_count = comp_df[comp_df['text'].str.contains('|'.join(failure_words), case=False, na=False)]
        
        if len(failure_count) > 0:
            results.append({
                'component': component,
                'failure_mentions': len(failure_count),
                'percentage': round(len(failure_count) / len(negative_df) * 100, 2) if len(negative_df) > 0 else 0
            })
    
    return pd.DataFrame(results).sort_values('failure_mentions', ascending=False)


def analyze_top_products(df):
    """Analyze sentiment breakdown by product"""
    if 'ProductName' not in df.columns or df['ProductName'].isna().all():
        return pd.DataFrame()
    
    product_sentiment = df.groupby(['ProductName', 'sentiment']).size().unstack(fill_value=0)
    product_sentiment['total'] = product_sentiment.sum(axis=1)
    product_sentiment = product_sentiment.sort_values('total', ascending=False).head(10)
    
    return product_sentiment.reset_index()


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'running',
        'message': 'E-commerce Sentiment Analysis API',
        'endpoints': {
            '/analyze': 'POST - Full ML pipeline analysis',
            '/outputs/<filename>': 'GET - Download generated CSV outputs',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


@app.route('/outputs/<path:filename>', methods=['GET'])
def download_output(filename: str):
    """Serve generated CSV outputs so the frontend can fetch them."""
    return send_from_directory(OUTPUTS_DIR, filename, as_attachment=False)

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Full ML pipeline:
    1. Column analysis (identify columns)
    2. Data cleaning
    3. ML predictions
    4. Keyword extraction
    5. Aspect sentiment analysis
    6. Component failure analysis
    7. Top products breakdown
    """
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save uploaded file
        input_path = os.path.join(DATA_DIR, 'uploaded.csv')
        try:
            file.save(input_path)
            print(f"File saved successfully to: {input_path}")
        except Exception as e:
            print(f"ERROR saving file: {str(e)}")
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
        
        # Read CSV with UTF-8 and replace invalid bytes
        try:
            df = pd.read_csv(input_path, encoding='utf-8', errors='replace')
            print(f"Successfully read CSV with encoding: utf-8 (errors='replace')")
        except Exception as e:
            print(f"ERROR reading CSV: {str(e)}")
            return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 500
        
        # Step 1: Column Analysis
        print("Step 1: Analyzing columns...")
        try:
            column_names = df.columns.tolist()
            first_rows = df.head(5).values.tolist()
            print(f"Columns: {column_names[:5]}...")
            
            column_mapping = analyze_columns_with_llm(column_names, first_rows)
            print(f"Column mapping received: {list(column_mapping.keys())}")
            
            # Save column mapping
            mapping_path = os.path.join(os.path.dirname(BASE_DIR), 'column_mapping.json')
            with open(mapping_path, 'w') as f:
                json.dump(column_mapping, f, indent=2)
            print(f"Column mapping saved to: {mapping_path}")
        except Exception as e:
            print(f"ERROR in column analysis: {str(e)}")
            return jsonify({'error': f'Column analysis failed: {str(e)}'}), 500
        
        # Step 2: Clean Data
        print("Step 2: Cleaning data...")
        cleaned_df = clean_data_pipeline(df, column_mapping)
        cleaned_path = os.path.join(DATA_DIR, 'uploaded_cleaned.csv')
        cleaned_df.to_csv(cleaned_path, index=False)
        
        # Step 3: Generate ML Predictions
        print("Step 3: Generating predictions...")
        predictions_df = generate_ml_predictions(cleaned_df)
        predictions_path = os.path.join(DATA_DIR, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        
        # Step 4: Extract Keywords
        print("Step 4: Extracting keywords...")
        positive_keywords, negative_keywords = extract_keywords_from_df(predictions_df)
        
        # Save keywords
        pos_path = os.path.join(OUTPUTS_DIR, 'positive_keywords.csv')
        neg_path = os.path.join(OUTPUTS_DIR, 'negative_keywords.csv')
        positive_keywords.to_csv(pos_path, index=False)
        negative_keywords.to_csv(neg_path, index=False)
        
        # Step 5: Aspect Sentiment Analysis
        print("Step 5: Analyzing aspect sentiment...")
        aspect_summary = analyze_aspects(predictions_df)
        
        aspect_summary_path = os.path.join(OUTPUTS_DIR, 'aspect_sentiment_summary.csv')
        aspect_summary.to_csv(aspect_summary_path, index=False)
        
        # Step 6: Component Failure Analysis
        print("Step 6: Analyzing component failures...")
        failure_components = analyze_component_failures(predictions_df)
        
        failure_path = os.path.join(OUTPUTS_DIR, 'failure_components_analysis.csv')
        failure_components.to_csv(failure_path, index=False)
        
        # Step 7: Top Products Breakdown
        print("Step 7: Analyzing top products...")
        top_products = analyze_top_products(predictions_df)
        
        products_path = os.path.join(OUTPUTS_DIR, 'top_products_sentiment_breakdown.csv')
        top_products.to_csv(products_path, index=False)
        
        # Return results
        return jsonify({
            'status': 'success',
            'message': 'Analysis completed successfully',
            'results': {
                'total_reviews': len(df),
                'sentiment_distribution': predictions_df['sentiment'].value_counts().to_dict(),
                'positive_keywords': positive_keywords.head(10).to_dict('records') if not positive_keywords.empty else [],
                'negative_keywords': negative_keywords.head(10).to_dict('records') if not negative_keywords.empty else [],
                'aspect_sentiment': aspect_summary.to_dict('records') if not aspect_summary.empty else [],
                'failure_components': failure_components.to_dict('records') if not failure_components.empty else [],
                'top_products': top_products.to_dict('records') if not top_products.empty else []
            },
            'output_files': {
                'predictions': 'predictions.csv',
                'positive_keywords': 'positive_keywords.csv',
                'negative_keywords': 'negative_keywords.csv',
                'aspect_sentiment_summary': 'aspect_sentiment_summary.csv',
                'failure_components': 'failure_components_analysis.csv',
                'top_products': 'top_products_sentiment_breakdown.csv'
            }
        })
        
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"ERROR in /analyze endpoint: {error_type}: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': error_msg,
            'type': error_type,
            'details': 'Check server logs for full traceback'
        }), 500

# ==========================================
# MODEL COMPARISON ENDPOINT
# ==========================================
@app.route('/compare', methods=['GET', 'POST'])
def compare_models():
    """
    Compare VADER vs TF-IDF + Logistic Regression models
    Returns accuracy metrics and LLM-formatted analysis
    """
    try:
        # Load comparison data from outputs if it exists
        metrics_file = os.path.join(OUTPUTS_DIR, 'model_comparison_metrics.json')
        report_file = os.path.join(OUTPUTS_DIR, 'model_comparison_report.txt')
        
        if not os.path.exists(metrics_file):
            # Run evaluation if not cached
            return run_model_comparison()
        
        # Load cached results
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        report = ""
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                report = f.read()
        
        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'report': report,
            'cached': True
        })
    
    except Exception as e:
        print(f"Error in compare_models: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

def run_model_comparison():
    """Run VADER vs Logistic Regression comparison"""
    try:
        # Load cleaned data
        csv_files = [f for f in os.listdir(DATA_DIR) if 'cleaned' in f.lower() and f.endswith('.csv')]
        if not csv_files:
            csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        
        if not csv_files:
            return jsonify({'error': 'No data files found'}), 400
        
        data_path = os.path.join(DATA_DIR, csv_files[0])
        
        # Read CSV with UTF-8 and replace invalid bytes
        df = pd.read_csv(data_path, encoding='utf-8', errors='replace')
        print(f"Read comparison data with utf-8 (errors='replace')")
        
        # Prepare data
        TEXT_COLUMN = "text"
        LABEL_COLUMN = "sentiment"
        
        df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
        
        # Map sentiment labels
        sentiment_map = {"positive": 1, "negative": 0, "neutral": 2}
        if LABEL_COLUMN in df.columns:
            df['sentiment_numeric'] = df[LABEL_COLUMN].map(lambda x: sentiment_map.get(str(x).lower(), -1))
        else:
            return jsonify({'error': f'{LABEL_COLUMN} column not found'}), 400
        
        X = df[TEXT_COLUMN]
        y = df['sentiment_numeric']
        
        # Split data
        class_counts = y.value_counts()
        stratify = None
        if (class_counts >= 2).all():
            stratify = y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        
        # Evaluate VADER
        sia = SentimentIntensityAnalyzer()
        vader_predictions = []
        
        for text in X_test:
            scores = sia.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.05:
                label = sentiment_map['positive']
            elif compound <= -0.05:
                label = sentiment_map['negative']
            else:
                label = sentiment_map['neutral']
            
            vader_predictions.append(label)
        
        vader_predictions = np.array(vader_predictions)
        
        vader_metrics = {
            'accuracy': float(accuracy_score(y_test, vader_predictions)),
            'precision': float(precision_score(y_test, vader_predictions, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, vader_predictions, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, vader_predictions, average='weighted', zero_division=0))
        }
        
        # Evaluate Logistic Regression
        try:
            tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
            model = joblib.load(os.path.join(MODELS_DIR, "logistic_model.joblib"))
            
            X_test_tfidf = tfidf.transform(X_test)
            lr_predictions = model.predict(X_test_tfidf)
            
            # Convert string predictions to numeric if necessary
            if isinstance(lr_predictions[0], str):
                lr_predictions_numeric = np.array([sentiment_map.get(str(p).lower(), -1) for p in lr_predictions])
            else:
                lr_predictions_numeric = lr_predictions
            
            lr_metrics = {
                'accuracy': float(accuracy_score(y_test, lr_predictions_numeric)),
                'precision': float(precision_score(y_test, lr_predictions_numeric, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, lr_predictions_numeric, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_test, lr_predictions_numeric, average='weighted', zero_division=0))
            }
        except FileNotFoundError:
            return jsonify({'error': 'Model files not found'}), 400
        
        # Calculate agreement
        agreement = (vader_predictions == lr_predictions_numeric).sum() / len(y_test) * 100
        
        comparison = {
            'vader': vader_metrics,
            'logistic_regression': lr_metrics,
            'comparison': {
                'agreement_percent': float(agreement),
                'test_size': int(len(y_test))
            }
        }
        
        # Generate report
        report = generate_comparison_report(vader_metrics, lr_metrics, agreement, len(y_test))
        
        # Save results
        metrics_file = os.path.join(OUTPUTS_DIR, 'model_comparison_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        report_file = os.path.join(OUTPUTS_DIR, 'model_comparison_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        return jsonify({
            'status': 'success',
            'metrics': comparison,
            'report': report,
            'cached': False
        })
    
    except Exception as e:
        print(f"Error in run_model_comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def generate_comparison_report(vader_metrics, lr_metrics, agreement, test_size):
    """Generate comparison report using LLM or default template"""
    
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

Model Agreement Rate: {agreement:.2f}%
Test Set Size: {test_size} samples

Please provide:
1. Key findings and model performance comparison
2. Strengths and weaknesses of each approach
3. Recommendations for which model to use in production
4. Suggestions for improvement

Format as a clear, professional report.
"""
    
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
        except Exception as e:
            print(f"OpenRouter error: {e}")
    
    # Fallback to default report
    return f"""
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
Agreement Rate: {agreement:.2f}%
Test Set Size: {test_size} samples

WINNER: {'TF-IDF + Logistic Regression' if lr_metrics['accuracy'] > vader_metrics['accuracy'] else 'VADER' if vader_metrics['accuracy'] > lr_metrics['accuracy'] else 'TIE'}
Accuracy Difference: {abs(lr_metrics['accuracy'] - vader_metrics['accuracy']):.4f}

RECOMMENDATIONS
{'-'*60}
Use {'TF-IDF + Logistic Regression' if lr_metrics['accuracy'] > vader_metrics['accuracy'] else 'VADER'} for production deployment based on accuracy metrics.
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
