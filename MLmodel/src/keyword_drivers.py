import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ----------------------------
# CONFIG
# ----------------------------
TEXT_COLUMN = "text"
POS_LABEL = "Positive"
NEG_LABEL = "Negative"
TOP_N = 20
EPS = 1e-9
DOC_FREQ_MIN = 10  # filter out ultra-rare typo-ish tokens
MIN_REVIEWS_PER_PRODUCT = 30
STOP_WORDS = ENGLISH_STOP_WORDS.union({"very", "really", "quite", "product", "item"})

BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BASE_DIR.parent
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = WORKSPACE_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ----------------------------
# HELPERS
# ----------------------------
def load_data():
    df = pd.read_csv(PREDICTIONS_PATH)
    df = df[df["sentiment_pred"].isin([POS_LABEL, NEG_LABEL])].copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    return df


def compute_mean_tfidf(texts, tfidf):
    vectors = tfidf.transform(texts)
    n_docs = vectors.shape[0]

    means = np.asarray(vectors.mean(axis=0)).ravel()
    doc_freq = np.asarray((vectors > 0).sum(axis=0)).ravel()
    doc_freq_pct = doc_freq / max(n_docs, 1)

    adjusted = means * np.log1p(doc_freq)

    feature_names = np.array(tfidf.get_feature_names_out())
    return feature_names, adjusted, means, doc_freq, doc_freq_pct


def top_keywords(feature_names, scores, means, doc_freq, doc_freq_pct, top_n=TOP_N):
    df = pd.DataFrame({
        "word": feature_names,
        "score_adj_mean_over_df": scores,
        "mean_tfidf": means,
        "doc_freq": doc_freq,
        "doc_freq_pct": doc_freq_pct,
    })

    df = df[df["doc_freq"] >= DOC_FREQ_MIN]
    df = df[df["word"].str.fullmatch(r"[a-z]{3,}")]  # keep alphabetic unigrams, drop noise/bi-grams
    df = df[~df["word"].isin(STOP_WORDS)]
    df = df.sort_values("score_adj_mean_over_df", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def get_product_keywords(product_name, df, tfidf, top_n=15):
    df_product = df[df["ProductName"] == product_name]
    if len(df_product) < MIN_REVIEWS_PER_PRODUCT:
        return {"message": "Not enough data for this product"}

    pos_texts = df_product.loc[df_product["sentiment_pred"] == POS_LABEL, TEXT_COLUMN]
    neg_texts = df_product.loc[df_product["sentiment_pred"] == NEG_LABEL, TEXT_COLUMN]

    if len(pos_texts) == 0 or len(neg_texts) == 0:
        return {"message": "Insufficient positive or negative reviews"}

    feature_names_pos, scores_pos, means_pos, dfreq_pos, dfreq_pct_pos = compute_mean_tfidf(pos_texts, tfidf)
    feature_names_neg, scores_neg, means_neg, dfreq_neg, dfreq_pct_neg = compute_mean_tfidf(neg_texts, tfidf)

    pos_df = top_keywords(feature_names_pos, scores_pos, means_pos, dfreq_pos, dfreq_pct_pos, top_n=top_n)
    neg_df = top_keywords(feature_names_neg, scores_neg, means_neg, dfreq_neg, dfreq_pct_neg, top_n=top_n)

    return {
        "positive_keywords": pos_df["word"].tolist(),
        "negative_keywords": neg_df["word"].tolist(),
    }


def plot_bar(df, title, path, value_col="score_adj_mean_over_df"):
    plt.figure(figsize=(8, 6))
    plt.barh(df["word"], df[value_col], color="#4e79a7")
    plt.title(title)
    plt.xlabel(value_col)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------
# MAIN
# ----------------------------
def main():
    tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")

    df = load_data()
    pos_texts = df.loc[df["sentiment_pred"] == POS_LABEL, TEXT_COLUMN]
    neg_texts = df.loc[df["sentiment_pred"] == NEG_LABEL, TEXT_COLUMN]

    feature_names_pos, scores_pos, means_pos, dfreq_pos, dfreq_pct_pos = compute_mean_tfidf(pos_texts, tfidf)
    feature_names_neg, scores_neg, means_neg, dfreq_neg, dfreq_pct_neg = compute_mean_tfidf(neg_texts, tfidf)

    pos_df = top_keywords(feature_names_pos, scores_pos, means_pos, dfreq_pos, dfreq_pct_pos)
    neg_df = top_keywords(feature_names_neg, scores_neg, means_neg, dfreq_neg, dfreq_pct_neg)

    pos_csv = OUTPUT_DIR / "positive_keywords.csv"
    neg_csv = OUTPUT_DIR / "negative_keywords.csv"
    pos_df.to_csv(pos_csv, index=False)
    neg_df.to_csv(neg_csv, index=False)

    plot_bar(pos_df, "Top Positive Keywords", OUTPUT_DIR / "positive_keywords.png")
    plot_bar(neg_df, "Top Negative Keywords", OUTPUT_DIR / "negative_keywords.png")

    print(f"Saved: {pos_csv}")
    print(f"Saved: {neg_csv}")
    print("Positive top words:\n", pos_df.head())
    print("Negative top words:\n", neg_df.head())


if __name__ == "__main__":
    main()
