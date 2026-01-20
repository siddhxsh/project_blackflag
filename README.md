# Project BlackFlag — Review Radar ⚡️

> Transform raw e-commerce reviews into actionable insights. Upload your data, get instant sentiment analysis, keyword drivers, and AI-powered summaries — no setup required.

[![Live Demo](https://img.shields.io/badge/Demo-Coming%20Soon-orange.svg)](#)
[![Status](https://img.shields.io/badge/Status-In%20Development-blue.svg)](#)

---

## 🎯 What is Review Radar?

**Review Radar** is a web-based sentiment analysis platform that turns messy e-commerce reviews (Amazon, Flipkart, etc.) into executive-ready insights. Just upload your CSV, and let the platform handle the rest.

Perfect for:
- 📊 **Product Managers** — Understand what customers love and hate
- 🛍️ **E-commerce Teams** — Monitor product performance at scale
- 📈 **Data Analysts** — Skip the cleaning, jump straight to insights
- 💼 **Business Leaders** — Get AI-powered executive summaries in seconds

---

## ✨ Features

### 🚿 **Intelligent Data Cleaning**
- Automatically handles messy CSVs with duplicate columns
- Normalizes text, prices, and ratings
- Robust handling of missing data

### 😊 **Advanced Sentiment Analysis**
- **Dual-model approach**: TF-IDF + Logistic Regression (primary) with VADER fallback
- Accurate sentiment classification: Positive, Negative, Neutral
- Compare model performance side-by-side

### 🔑 **Keyword Intelligence**
- Extracts top positive and negative sentiment drivers
- Ranked by statistical significance (TF-IDF)
- Discover what actually matters to your customers

### 🧭 **Aspect-Based Insights**
Break down sentiment by key dimensions:
- 💰 **Price** — Value perception
- ⭐ **Quality** — Product satisfaction
- 🚚 **Delivery** — Logistics experience

### 🔧 **Failure Detection**
- Automatically surfaces hardware issues and component failures
- Identifies recurring problem patterns
- Perfect for electronics and physical product reviews

### 🏆 **Product Ranking**
- Compare sentiment across multiple products
- Keyword highlights for each product
- Identify your stars and problem children

### 🧠 **AI Executive Summaries**
- One-click summary generation powered by LLMs
- Distills thousands of reviews into key takeaways
- Perfect for reports and presentations

### ⚖️ **Model Transparency**
- Side-by-side comparison of VADER vs TF-IDF+LogReg
- View precision, recall, and F1 scores
- Understand the confidence behind predictions

---

## 🚀 How It Works

```
1️⃣  Upload Your CSV
     ↓
2️⃣  Platform Cleans & Validates Data
     ↓
3️⃣  Sentiment Analysis Runs Automatically
     ↓
4️⃣  Keywords, Aspects & Failures Extracted
     ↓
5️⃣  Download Results or Generate AI Summary
```

**Time to insights:** Under 1 minute for most datasets ⚡

---

## 📊 What You Get

### Instant Downloads

| Output | What's Inside |
|--------|---------------|
| **Predictions Report** | Every review with sentiment labels |
| **Positive Keywords** | Top drivers of customer satisfaction |
| **Negative Keywords** | Top drivers of customer complaints |
| **Aspect Breakdown** | Sentiment by Price, Quality, Delivery |
| **Failure Analysis** | Component issues and patterns |
| **Product Comparison** | Sentiment rankings across products |
| **Model Metrics** | Performance comparison (JSON + TXT) |

### AI-Powered Summary
Get a concise executive summary covering:
- Overall sentiment trends
- Key strengths and weaknesses
- Actionable recommendations
- Critical issues requiring attention

---

## 🎨 Platform Preview

### Web Interface Features
- 📤 **Drag-and-drop CSV upload** — No technical knowledge required
- 🎯 **Column mapping** — Tell us where your reviews and ratings are
- 📈 **Real-time progress** — Watch your analysis in action
- 📥 **One-click downloads** — Get all outputs in CSV format
- 🤖 **Summary generator** — AI insights on demand
- 🔄 **Model comparison** — See how different models perform

---

## 📁 Expected CSV Format

Your CSV should contain:
- **Review text column** (required) — The actual review content
- **Rating column** (optional) — Numeric ratings (1-5 stars)
- **Product identifier** (optional) — For product-level comparisons

**Example:**
```csv
review_text,rating,product_id
"Great product, fast delivery!",5,PROD001
"Terrible quality, broke in a week",1,PROD002
"Okay for the price",3,PROD001
```

> 💡 **Don't worry about formatting** — Review Radar automatically detects and cleans common issues.

---

## 🛠️ Behind the Scenes

### The Technology
- **Backend**: FastAPI (Python) — Fast, reliable API
- **ML Models**: Dual sentiment analysis with TF-IDF + Logistic Regression and VADER
- **NLP Pipeline**: NLTK and scikit-learn for text processing
- **AI Summaries**: Powered by Gemini AI and OpenRouter
- **Data Processing**: Pandas and NumPy for robust data handling

### Why It's Reliable
- ✅ **Production-grade** — Timeout guards and thread-safe operations
- ✅ **Fault-tolerant** — Graceful handling of missing data
- ✅ **Deterministic** — Same input = same output, every time
- ✅ **Scalable** — Handles datasets from hundreds to thousands of reviews

---

## 🌐 Access

**Live Application:** Coming Soon

**API Documentation:** Available at `/docs` once live

**Status Updates:** Watch this repo for launch announcements

---

## 💡 Use Cases

### 📦 Product Launch Analysis
Upload reviews from your latest product launch to understand initial reception.

### 🔍 Competitive Intelligence
Compare sentiment across competitor products on Amazon or Flipkart.

### 📅 Trend Monitoring
Track sentiment changes over time by analyzing reviews by date range.

### 🚨 Crisis Detection
Quickly identify emerging issues through failure detection and negative keyword analysis.

### 📊 Quarterly Reports
Generate executive summaries for stakeholder presentations.

---

## 🤝 Contributing

Found a bug? Have a feature request? Contributions are welcome!

- 🐛 **Report bugs** — Open an issue
- 💡 **Suggest features** — Start a discussion
- 🔧 **Submit PRs** — Check open issues for ways to contribute

---

## 📧 Contact & Support

**Developer:** Siddharth ([@siddhxsh](https://github.com/siddhxsh))

**Project Repository:** [github.com/siddhxsh/project_blackflag](https://github.com/siddhxsh/project_blackflag)

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

Built with [FastAPI](https://fastapi.tiangolo.com/), [NLTK](https://www.nltk.org/), [scikit-learn](https://scikit-learn.org/), and AI models from [Google Gemini](https://ai.google.dev/) and [OpenRouter](https://openrouter.ai/).

---

⭐️ **Star this repo to get notified when Review Radar goes live!**