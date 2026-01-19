## Project BlackFlag — Review Radar ⚡️

Pull raw e-com reviews (Amazon, Flipkart, etc.), clean them, score sentiment, surface drivers/aspects/failures, and spit out exec-ready summaries via API.

## What it does
- 🚿 Cleans messy CSVs (dupe columns, text normalize, price/rate coercion)
- 😊 Sentiment via TF-IDF + Logistic Regression (VADER as fallback)
- 🔑 Keyword drivers (pos/neg) ranked by TF-IDF stats
- 🧭 Aspect sentiment for Price / Quality / Delivery
- 🔧 Failure/component surfacing for hardware-ish issues
- 🏆 Top-product sentiment + keyword highlights
- 🧠 Exec summaries via `/formulate` (Gemini or OpenRouter)
- ⚖️ Model face-off: VADER vs TF-IDF+LogReg

## Flow in 5 steps
1) `/analyze` → map columns, clean, validate
2) Predict → sentiments saved to `outputs/`
3) Enrich → keywords, aspects, failures, top products
4) `/formulate` → summary from cached outputs
5) `/compare` → serve or compute model metrics

## Why it holds up
- Deterministic cleaning + dupe-column resolution
- Soft landings when cols/models are missing; solid NaN handling
- NLTK bits auto-fetched for prod boxes
- Timeout + thread-safe signal guards for hosts

## API menu
- `/health` — liveness ping
- `/analyze` — full pipeline on upload
- `/formulate` — LLM summary from cache
- `/compare` — VADER vs TF-IDF+LogReg metrics/report

## Outputs
- `outputs/predictions.csv` — cleaned text + sentiment
- `outputs/positive_keywords.csv` / `outputs/negative_keywords.csv`
- `outputs/aspect_sentiment_summary.csv`
- `outputs/failure_components_analysis.csv`
- `outputs/top_products_sentiment_breakdown.csv`
- `outputs/model_comparison_metrics.json` and `.txt`

## Frontend (placeholder)
UI will hook these APIs to visualize sentiments, aspects, keywords, and summaries. (Teammate will wire it up.)

