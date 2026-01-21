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
# CONFIG (env-overridable thresholds)
# ----------------------------
TOP_N_PRODUCTS = int(os.getenv("TOP_N_PRODUCTS", "10"))
TOP_N_KEYWORDS = int(os.getenv("TOP_N_KEYWORDS", "5"))  # top keywords per sentiment per product
MIN_REVIEWS_PER_PRODUCT = int(os.getenv("MIN_REVIEWS_PER_PRODUCT", "30"))
STOP_WORDS = ENGLISH_STOP_WORDS.union({"very", "really", "quite", "product", "item"})

BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BASE_DIR.parent
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = WORKSPACE_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ----------------------------
# HELPERS (from keyword_drivers.py)
# ----------------------------
def compute_mean_tfidf(texts, tfidf):
    vectors = tfidf.transform(texts)
    n_docs = vectors.shape[0]
    means = np.asarray(vectors.mean(axis=0)).ravel()
    doc_freq = np.asarray((vectors > 0).sum(axis=0)).ravel()
    doc_freq_pct = doc_freq / max(n_docs, 1)
    adjusted = means * np.log1p(doc_freq)
    feature_names = np.array(tfidf.get_feature_names_out())
    return feature_names, adjusted, means, doc_freq, doc_freq_pct


def top_keywords(feature_names, scores, means, doc_freq, doc_freq_pct, top_n=TOP_N_KEYWORDS):
    df = pd.DataFrame({
        "word": feature_names,
        "score_adj_mean_over_df": scores,
        "mean_tfidf": means,
        "doc_freq": doc_freq,
        "doc_freq_pct": doc_freq_pct,
    })
    df = df[df["doc_freq"] >= 10]  # DOC_FREQ_MIN
    df = df[df["word"].str.fullmatch(r"[a-z]{3,}")]  # alphabetic unigrams
    df = df[~df["word"].isin(STOP_WORDS)]
    df = df.sort_values("score_adj_mean_over_df", ascending=False).head(top_n)
    return df["word"].tolist()


def get_product_keywords(product_name, df, tfidf):
    df_product = df[df["ProductName"] == product_name]
    if len(df_product) < MIN_REVIEWS_PER_PRODUCT:
        return {"positive_keywords": [], "negative_keywords": []}

    pos_texts = df_product.loc[df_product["sentiment_pred"] == "Positive", "text"]
    neg_texts = df_product.loc[df_product["sentiment_pred"] == "Negative", "text"]

    pos_keywords = []
    neg_keywords = []

    if len(pos_texts) > 0:
        feature_names_pos, scores_pos, means_pos, dfreq_pos, dfreq_pct_pos = compute_mean_tfidf(pos_texts, tfidf)
        pos_keywords = top_keywords(feature_names_pos, scores_pos, means_pos, dfreq_pos, dfreq_pct_pos)

    if len(neg_texts) > 0:
        feature_names_neg, scores_neg, means_neg, dfreq_neg, dfreq_pct_neg = compute_mean_tfidf(neg_texts, tfidf)
        neg_keywords = top_keywords(feature_names_neg, scores_neg, means_neg, dfreq_neg, dfreq_pct_neg)

    return {
        "positive_keywords": pos_keywords,
        "negative_keywords": neg_keywords,
    }


def main():
    df = pd.read_csv(PREDICTIONS_PATH)
    tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")

    # Compute review counts per product
    product_counts = df.groupby("ProductName").size().reset_index(name="total_reviews")
    top_products = product_counts.sort_values("total_reviews", ascending=False).head(TOP_N_PRODUCTS)

    # Compute sentiment breakdown
    sentiment_breakdown = df[df["ProductName"].isin(top_products["ProductName"])].groupby("ProductName")["sentiment_pred"].value_counts().unstack(fill_value=0)
    sentiment_breakdown = sentiment_breakdown.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    sentiment_breakdown["Total"] = sentiment_breakdown.sum(axis=1)

    # Merge with counts
    top_products_breakdown = top_products.merge(sentiment_breakdown, on="ProductName")

    # Add keywords
    pos_keywords_list = []
    neg_keywords_list = []
    for product in top_products_breakdown["ProductName"]:
        keywords = get_product_keywords(product, df, tfidf)
        pos_keywords_list.append(", ".join(keywords["positive_keywords"]))
        neg_keywords_list.append(", ".join(keywords["negative_keywords"]))

    top_products_breakdown["top_positive_keywords"] = pos_keywords_list
    top_products_breakdown["top_negative_keywords"] = neg_keywords_list

    # Save CSV
    csv_path = OUTPUT_DIR / "top_products_sentiment_breakdown.csv"
    top_products_breakdown.to_csv(csv_path, index=False)

    # Create stacked bar chart (unchanged)
    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = [0] * len(top_products_breakdown)
    colors = ["#4CAF50", "#FFC107", "#F44336"]  # Positive, Neutral, Negative
    labels = ["Positive", "Neutral", "Negative"]

    for i, sentiment in enumerate(["Positive", "Neutral", "Negative"]):
        ax.bar(
            top_products_breakdown["ProductName"],
            top_products_breakdown[sentiment],
            bottom=bottom,
            label=labels[i],
            color=colors[i]
        )
        bottom = [b + v for b, v in zip(bottom, top_products_breakdown[sentiment])]

    ax.set_title("Top Reviewed Products — Sentiment Breakdown")
    ax.set_ylabel("Number of Reviews")
    ax.set_xlabel("Product Name")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    chart_path = OUTPUT_DIR / "top_products_sentiment_breakdown.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Chart: {chart_path}")
    print(top_products_breakdown)


if __name__ == "__main__":
    main()