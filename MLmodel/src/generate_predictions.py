import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from utils import find_cleaned_csv, get_base_dir, get_model_dir

# ----------------------------
# CONFIG
# ----------------------------
TEXT_COLUMN = "text"
RATE_COLUMN = "Rate"

BASE_DIR = get_base_dir()
MODEL_DIR = get_model_dir()


# ----------------------------
# PREPROCESSING
# ----------------------------
def clean_text(text: str) -> str:
    """Mirror training-time preprocessing."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_rate_to_sentiment(rate):
    """Optional ground-truth sentiment derived from numeric rating."""
    try:
        val = int(rate)
    except (TypeError, ValueError):
        return None

    if val >= 4:
        return "Positive"
    if val == 3:
        return "Neutral"
    if val in {1, 2}:
        return "Negative"
    return None


def resolve_text_series(df: pd.DataFrame) -> pd.Series:
    """Provide a text series; fallback to Review/Summary if needed."""
    if TEXT_COLUMN in df:
        return df[TEXT_COLUMN].fillna("").astype(str)

    # Common fallbacks if text column is missing
    if {"Summary", "Review"}.issubset(df.columns):
        combined = (df["Summary"].fillna("") + " " + df["Review"].fillna("")).str.strip()
        return combined
    if "Review" in df:
        return df["Review"].fillna("").astype(str)
    if "Summary" in df:
        return df["Summary"].fillna("").astype(str)

    raise KeyError(f"None of the expected text columns ({TEXT_COLUMN}, Review, Summary) were found.")


# ----------------------------
# CORE PIPELINE
# ----------------------------
def generate_predictions(input_path: Path, output_path: Path) -> Path:
    tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODEL_DIR / "logistic_model.joblib")

    df = pd.read_csv(input_path)
    text_series = resolve_text_series(df)

    cleaned = text_series.apply(clean_text)
    vectors = tfidf.transform(cleaned)
    preds = model.predict(vectors)

    output_df = pd.DataFrame(index=df.index)
    if "ProductName" in df.columns:
        output_df["ProductName"] = df["ProductName"]
    output_df[TEXT_COLUMN] = text_series
    if RATE_COLUMN in df.columns:
        output_df[RATE_COLUMN] = df[RATE_COLUMN]
    output_df["sentiment_pred"] = preds
    if RATE_COLUMN in df.columns:
        output_df["sentiment_true"] = df[RATE_COLUMN].apply(map_rate_to_sentiment)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return output_path


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    # Auto-detect cleaned CSV in data folder
    try:
        default_input = find_cleaned_csv(BASE_DIR / "data")
    except FileNotFoundError:
        default_input = BASE_DIR / "data" / "cleaned_data.csv"
    
    # Save predictions.csv in data folder
    default_output = BASE_DIR / "data" / "predictions.csv"

    parser = argparse.ArgumentParser(description="Generate sentiment predictions CSV.")
    parser.add_argument("--input", type=Path, default=default_input, help="Path to cleaned CSV (default: first *_cleaned.csv in data folder)")
    parser.add_argument("--output", type=Path, default=default_output, help="Where to save predictions CSV (default: data folder)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_file = generate_predictions(args.input, args.output)
    print(f"Predictions written to: {output_file}")


if __name__ == "__main__":
    main()
