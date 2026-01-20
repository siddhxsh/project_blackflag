import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import joblib
import nltk
import pandas as pd

# Ensure sentence tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize

# ----------------------------
# CONFIG
# ----------------------------
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "Price": ["price", "cost", "expensive", "cheap", "worth"],
    "Quality": ["quality", "build", "material", "durable", "poor"],
    "Delivery": ["delivery", "shipping", "late", "fast", "delay"],
}

TEXT_COLUMN = "text"
BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BASE_DIR.parent
MODEL_DIR = BASE_DIR / "models"
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.csv"
OUTPUT_DIR = WORKSPACE_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ----------------------------
# PREPROCESSING
# ----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resolve_predictions_path() -> Path:
    if PREDICTIONS_PATH.exists():
        return PREDICTIONS_PATH
    raise FileNotFoundError(f"predictions.csv not found at {PREDICTIONS_PATH}")


# ----------------------------
# CORE LOGIC
# ----------------------------
def predict_sentence_sentiments(sentences: List[str], tfidf, model) -> List[str]:
    if not sentences:
        return []
    cleaned = [clean_text(s) for s in sentences]
    vectors = tfidf.transform(cleaned)
    return model.predict(vectors)


def majority_label(labels: List[str]) -> str:
    if not labels:
        return "Not Mentioned"
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def process_aspects(df: pd.DataFrame, tfidf, model) -> pd.DataFrame:
    aspect_sentiments = {aspect: [] for aspect in ASPECT_KEYWORDS}
    aspect_example_sentences = {aspect: [] for aspect in ASPECT_KEYWORDS}

    for _, row in df.iterrows():
        text = str(row.get(TEXT_COLUMN, ""))
        sentences = sent_tokenize(text)

        row_aspect_sentiments = {}
        row_aspect_examples = {}

        for aspect, keywords in ASPECT_KEYWORDS.items():
            matched_sents = []
            for sent in sentences:
                sent_lower = sent.lower()
                if any(k in sent_lower for k in keywords):
                    matched_sents.append(sent)

            preds = predict_sentence_sentiments(matched_sents, tfidf, model)
            label = majority_label(list(preds))

            row_aspect_sentiments[aspect] = label
            row_aspect_examples[aspect] = " | ".join(matched_sents) if matched_sents else ""

        for aspect in ASPECT_KEYWORDS:
            aspect_sentiments[aspect].append(row_aspect_sentiments[aspect])
            aspect_example_sentences[aspect].append(row_aspect_examples[aspect])

    enriched = df.copy()
    for aspect in ASPECT_KEYWORDS:
        enriched[f"aspect_{aspect}_sentiment"] = aspect_sentiments[aspect]
        enriched[f"aspect_{aspect}_sentences"] = aspect_example_sentences[aspect]

    return enriched


def build_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for aspect in ASPECT_KEYWORDS:
        subset = enriched[f"aspect_{aspect}_sentiment"].value_counts().reindex(
            ["Positive", "Neutral", "Negative", "Not Mentioned"], fill_value=0
        )
        rows.append({
            "aspect": aspect,
            "Positive": subset["Positive"],
            "Neutral": subset["Neutral"],
            "Negative": subset["Negative"],
            "Not Mentioned": subset["Not Mentioned"],
        })
    return pd.DataFrame(rows)


# ----------------------------
# ENTRYPOINT
# ----------------------------
def main():
    predictions_path = resolve_predictions_path()
    df = pd.read_csv(predictions_path)

    tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODEL_DIR / "logistic_model.joblib")

    enriched = process_aspects(df, tfidf, model)

    summary = build_summary(enriched)
    summary_path = OUTPUT_DIR / "aspect_sentiment_summary.csv"
    summary.to_csv(summary_path, index=False)

    examples_path = OUTPUT_DIR / "aspect_sentiment_examples.csv"

    aspect_cols = [f"aspect_{a}_sentiment" for a in ASPECT_KEYWORDS]
    mention_mask = (enriched[aspect_cols] != "Not Mentioned").any(axis=1)

    cols_to_keep = [
        c for c in [
            "ProductName",
            "sentiment_pred",
            "Rate",
            TEXT_COLUMN,
        ] if c in enriched.columns
    ]
    cols_to_keep += aspect_cols
    cols_to_keep += [f"aspect_{a}_sentences" for a in ASPECT_KEYWORDS]

    tidy_examples = enriched.loc[mention_mask, cols_to_keep].head(100)
    tidy_examples.to_csv(examples_path, index=False)

    print(f"Summary saved to {summary_path}")
    print(f"Examples saved to {examples_path}")
    print(summary)


if __name__ == "__main__":
    main()
