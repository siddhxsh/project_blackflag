# E-Commerce Sentiment Analysis API

Live API: **https://project-blackflag.onrender.com**

A comprehensive Flask-based API for analyzing e-commerce product reviews using machine learning and NLP techniques. The API provides sentiment analysis, keyword extraction, aspect sentiment analysis, component failure detection, and more.

---

## 📋 API Endpoints

### 1. **Health Check** - Verify API Status
```bash
GET /health
```

**cURL Example:**
```bash
curl -X GET https://project-blackflag.onrender.com/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

---

### 2. **Home** - View Available Endpoints
```bash
GET /
```

**cURL Example:**
```bash
curl -X GET https://project-blackflag.onrender.com/
```

**Response:**
```json
{
  "status": "running",
  "message": "E-commerce Sentiment Analysis API",
  "endpoints": {
    "/analyze": "POST - Full ML pipeline analysis",
    "/outputs/<filename>": "GET - Download generated CSV outputs",
    "/health": "GET - Health check",
    "/formulate": "POST - Generate executive summary using LLM",
    "/compare": "GET/POST - Compare VADER vs Logistic Regression models"
  }
}
```

---

### 3. **Analyze Data** - Full ML Pipeline Analysis
```bash
POST /analyze
```

**Description:** Upload a CSV file for complete sentiment analysis including:
- Column analysis and mapping
- Data cleaning
- ML predictions (TF-IDF + Logistic Regression)
- Keyword extraction (positive & negative)
- Aspect sentiment analysis
- Component failure analysis
- Top products breakdown

**Required:**
- File upload: CSV file with product reviews

**Supported CSV Columns:**
- `ProductName` - Name of the product
- `Price` - Product price
- `Rate` - Rating (1-5 stars)
- `Review` - Review text
- `Summary` - Review summary

> **Note:** Column names are auto-detected. You can use different column names and the API will map them intelligently.

**cURL Example:**
```bash
curl -X POST https://project-blackflag.onrender.com/analyze \
  -F "file=@your_reviews.csv"
```

**Python Example:**
```python
import requests

with open('your_reviews.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'https://project-blackflag.onrender.com/analyze',
        files=files
    )
    result = response.json()
    print(result)
```

**Response:**
```json
{
  "status": "success",
  "message": "Analysis completed successfully",
  "summary": {
    "total_reviews": 1000,
    "sentiment_summary": {
      "positive": 650,
      "negative": 200,
      "neutral": 150
    }
  },
  "data": {
    "positive_keywords": [...],
    "negative_keywords": [...],
    "aspect_sentiment": [...],
    "failure_components": [...],
    "top_products": [...]
  },
  "output_files": {
    "predictions": "predictions.csv",
    "positive_keywords": "positive_keywords.csv",
    "negative_keywords": "negative_keywords.csv",
    "aspect_sentiment_summary": "aspect_sentiment_summary.csv",
    "failure_components": "failure_components_analysis.csv",
    "top_products": "top_products_sentiment_breakdown.csv"
  }
}
```

---

### 4. **Download Output Files**
```bash
GET /outputs/<filename>
```

**Description:** Download any of the generated CSV analysis files from the `/analyze` endpoint.

**Available Files:**
- `predictions.csv` - All reviews with predicted sentiment
- `positive_keywords.csv` - Top positive keywords with TF-IDF scores
- `negative_keywords.csv` - Top negative keywords with TF-IDF scores
- `aspect_sentiment_summary.csv` - Aspect-based sentiment breakdown
- `failure_components_analysis.csv` - Component failures by product
- `top_products_sentiment_breakdown.csv` - Top 10 products with sentiment breakdown

**cURL Example:**
```bash
curl -X GET https://project-blackflag.onrender.com/outputs/predictions.csv -o predictions.csv
```

**Python Example:**
```python
import requests

response = requests.get('https://project-blackflag.onrender.com/outputs/predictions.csv')
with open('predictions.csv', 'wb') as f:
    f.write(response.content)
```

---

### 5. **Generate Executive Summary** - LLM Analysis
```bash
POST /formulate
```

**Description:** Generate an AI-powered executive summary from analysis results using Google Gemini or OpenRouter.

**Request Body (JSON):**
```json
{
  "llm_provider": "gemini",
  "model": "gemini-2.0-flash-exp"
}
```

**Parameters:**
- `llm_provider` (string, optional): `"gemini"` or `"openrouter"` (default: `"gemini"`)
- `model` (string, optional): Specific model name to use

**cURL Example:**
```bash
curl -X POST https://project-blackflag.onrender.com/formulate \
  -H "Content-Type: application/json" \
  -d '{
    "llm_provider": "gemini",
    "model": "gemini-2.0-flash-exp"
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    'https://project-blackflag.onrender.com/formulate',
    json={
        'llm_provider': 'gemini',
        'model': 'gemini-2.0-flash-exp'
    }
)
result = response.json()
print(result['executive_summary'])
```

**Response:**
```json
{
  "status": "success",
  "llm_provider": "gemini",
  "total_reviews": 1000,
  "sentiment_summary": {
    "positive": 650,
    "negative": 200,
    "neutral": 150
  },
  "executive_summary": "## EXECUTIVE SUMMARY\n\n... [LLM-generated insights] ..."
}
```

---

### 6. **Compare Models** - VADER vs Logistic Regression
```bash
GET /compare
POST /compare
```

**Description:** Compare sentiment analysis performance between VADER (rule-based) and TF-IDF + Logistic Regression (ML-based) models.

**cURL Example:**
```bash
curl -X GET https://project-blackflag.onrender.com/compare
```

**Python Example:**
```python
import requests

response = requests.get('https://project-blackflag.onrender.com/compare')
result = response.json()
print(result['metrics'])
print(result['report'])
```

**Response:**
```json
{
  "status": "success",
  "metrics": {
    "vader": {
      "accuracy": 0.75,
      "precision": 0.73,
      "recall": 0.75,
      "f1": 0.74
    },
    "logistic_regression": {
      "accuracy": 0.82,
      "precision": 0.81,
      "recall": 0.82,
      "f1": 0.81
    },
    "comparison": {
      "agreement_percent": 78.5,
      "test_size": 200
    }
  },
  "report": "... [Model comparison analysis] ...",
  "cached": true
}
```

---

## 🚀 Quick Start Guide

### Step 1: Upload and Analyze Data
```bash
curl -X POST https://project-blackflag.onrender.com/analyze \
  -F "file=@amazon_reviews.csv"
```

### Step 2: Check Sentiment Distribution
The response will show:
- Total reviews processed
- Sentiment counts (Positive/Negative/Neutral)
- Top keywords for each sentiment
- Aspect-based analysis
- Top products breakdown
- Component failures

### Step 3: Download Detailed Results
```bash
curl -X GET https://project-blackflag.onrender.com/outputs/predictions.csv -o predictions.csv
curl -X GET https://project-blackflag.onrender.com/outputs/positive_keywords.csv -o positive_keywords.csv
curl -X GET https://project-blackflag.onrender.com/outputs/negative_keywords.csv -o negative_keywords.csv
```

### Step 4: Generate Executive Summary
```bash
curl -X POST https://project-blackflag.onrender.com/formulate \
  -H "Content-Type: application/json" \
  -d '{"llm_provider": "gemini"}'
```

---

## 📊 Expected CSV Format

Your input CSV should contain review data with columns like:
| ProductName | Price | Rate | Review | Summary |
|------------|-------|------|---------|---------|
| Product A | $99.99 | 5 | Great product! Works as expected. | Excellent |
| Product B | $49.99 | 2 | Poor quality, stopped working | Bad quality |

**Supported Column Names:**
- Product: `ProductName`, `product_name`, `product_title`, `product`, `title`, `name`
- Price: `price`, `cost`, `amount`, `mrp`
- Rating: `rating`, `rate`, `stars`, `score`
- Review: `review`, `comment`, `feedback`, `description`, `text`
- Summary: `summary`, `headline`, `subject`, `title`

---

## 📝 Response Data Structure

### Sentiment Prediction Fields:
- `predicted_sentiment`: ML model prediction (Positive/Negative/Neutral)
- `sentiment`: Rule-based sentiment from rating (1-2: Negative, 3: Neutral, 4-5: Positive)

### Keyword Fields:
- `word`: Keyword term
- `mean_tfidf`: Average TF-IDF score
- `doc_frequency`: Number of documents containing the word
- `doc_frequency_pct`: Percentage of documents

### Aspect Sentiment Fields:
- `aspect`: Product aspect (e.g., "quality", "durability")
- `Positive`: Count of positive mentions
- `Negative`: Count of negative mentions
- `Neutral`: Count of neutral mentions
- `Not Mentioned`: Count of reviews not mentioning aspect

### Top Products Fields:
- `ProductName`: Product name
- `ReviewCount`: Total reviews
- `Positive`: Count of positive reviews
- `Negative`: Count of negative reviews
- `Neutral`: Count of neutral reviews
- `TopPositiveKeywords`: Top keywords in positive reviews
- `TopNegativeKeywords`: Top keywords in negative reviews

---

## 🔧 Environment Variables

The API uses the following environment variables (if configuring locally):
```
GOOGLE_API_KEY=<your-google-gemini-api-key>
OPENROUTER_API_KEY=<your-openrouter-api-key>
LLM_MODEL=openai/gpt-4o (or other model)
PORT=5000
```

---

## ⚡ Performance Notes

- **Upload Limit:** Typically handles up to 50MB files
- **Processing Time:** 
  - Small files (< 1000 reviews): ~10-30 seconds
  - Medium files (1000-10000 reviews): ~30-60 seconds
  - Large files (> 10000 reviews): ~60+ seconds
- **Timeout:** 120 seconds for complete analysis

---

## 🛠️ Technologies Used

- **Flask** - REST API framework
- **Pandas** - Data processing
- **Scikit-learn** - Machine learning (TF-IDF vectorizer, Logistic Regression)
- **NLTK** - NLP and VADER sentiment analysis
- **Joblib** - Model persistence
- **Google Gemini API** - LLM for summaries
- **OpenRouter API** - Alternative LLM provider

---

## 📞 Support

For issues or questions:
1. Check API health: `GET /health`
2. Review error messages in the JSON response
3. Ensure CSV format matches requirements
4. Verify file is valid UTF-8 encoded

---

## 📄 License

This project is part of Project BlackFlag.

---

**Last Updated:** January 2026  
**API Status:** Live on Render  
**Base URL:** https://project-blackflag.onrender.com
