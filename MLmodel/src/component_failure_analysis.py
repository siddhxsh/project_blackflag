import os
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk import pos_tag, word_tokenize

# Download NLTK tokenizer and POS tagger if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# ----------------------------
# CONFIG (env-overridable thresholds)
# ----------------------------
MIN_NEGATIVE_REVIEWS = int(os.getenv("MIN_NEGATIVE_REVIEWS", "15"))
TOP_N_COMPONENTS = int(os.getenv("TOP_N_COMPONENTS", "3"))  # Only top 3 for consolidated view
TOP_N_PRODUCTS_FOR_CHARTS = int(os.getenv("TOP_N_PRODUCTS_FOR_CHARTS", "3"))

GENERIC_NOUNS = {
    "product", "item", "thing", "time", "day", "month", "week", "year",
    "experience", "quality", "price", "value", "service", "reason",
}

RETAIL_WORDS = {"amazon", "flipkart", "seller", "shop", "store", "order"}

BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BASE_DIR.parent
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.csv"
OUTPUT_DIR = WORKSPACE_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ----------------------------
# HELPERS
# ----------------------------
def clean_text_for_pos(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    return text


def extract_nouns_and_verbs(text: str) -> list:
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    words = []
    for word, tag in pos_tags:
        # Include nouns (NN, NNS) and verbs (VB, VBD, VBG, VBN, VBP, VBZ)
        if tag in {"NN", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}:
            word_clean = re.sub(r"[^a-z]", "", word)
            if len(word_clean) >= 3 and word_clean not in GENERIC_NOUNS and word_clean not in RETAIL_WORDS:
                words.append(word_clean)

    return words


# ----------------------------
# MAIN
# ----------------------------
def analyze_product_failures(product_name, df):
    df_product = df[df["ProductName"] == product_name]
    if len(df_product) == 0:
        print(f"  No reviews found for: {product_name}")
        return None

    df_neg = df_product[df_product["sentiment_pred"] == "Negative"]
    if len(df_neg) < MIN_NEGATIVE_REVIEWS:
        print(f"  Too low negative reviews: {len(df_neg)} (need {MIN_NEGATIVE_REVIEWS})")
        return None
    
    print(f"  Found {len(df_neg)} negative reviews")

    all_words = []
    for text in df_neg.get("text", []):
        text = clean_text_for_pos(str(text))
        words = extract_nouns_and_verbs(text)
        all_words.extend(words)

    word_counts = Counter(all_words)
    top_words = word_counts.most_common(TOP_N_COMPONENTS)
    print(f"  Extracted {len(all_words)} total words, {len(word_counts)} unique")

    if not top_words:
        return None

    top_components = [word for word, freq in top_words]
    return top_components


if __name__ == "__main__":
    df = pd.read_csv(PREDICTIONS_PATH)
    print(f"Loaded {len(df)} predictions")
    top_products_path = OUTPUT_DIR / "top_products_sentiment_breakdown.csv"
    
    if top_products_path.exists():
        top_products_df = pd.read_csv(top_products_path)
        print(f"Found {len(top_products_df)} top products")
        
        results = []
        for idx, row in top_products_df.iterrows():
            product = row["ProductName"]
            components = analyze_product_failures(product, df)
            
            if components:
                results.append({
                    "product_rank": idx + 1,
                    "ProductName": product,
                    "top_component_1": components[0] if len(components) > 0 else "",
                    "top_component_2": components[1] if len(components) > 1 else "",
                    "top_component_3": components[2] if len(components) > 2 else "",
                })
                print(f"[{idx+1}] {product}")
                print(f"    Top 3: {', '.join(components)}")
        
        # Save consolidated CSV
        if results:
            result_df = pd.DataFrame(results)
            csv_path = OUTPUT_DIR / "failure_components_analysis.csv"
            result_df.to_csv(csv_path, index=False)
            print(f"\nSaved consolidated CSV: {csv_path}")
            
            # Create charts only for top 3 products
            for idx, row in result_df.head(TOP_N_PRODUCTS_FOR_CHARTS).iterrows():
                product = row["ProductName"]
                components = [row["top_component_1"], row["top_component_2"], row["top_component_3"]]
                components = [c for c in components if c]  # Remove empty strings
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(components, range(len(components), 0, -1), color="#e74c3c")
                ax.set_xlabel("Rank")
                ax.set_title(f"Top Problem Components — Product #{row['product_rank']}")
                ax.set_yticks(range(len(components)))
                ax.set_yticklabels(components)
                ax.invert_yaxis()
                plt.tight_layout()
                
                chart_path = OUTPUT_DIR / f"failure_components_product_{row['product_rank']}.png"
                plt.savefig(chart_path, dpi=150)
                plt.close()
                print(f"Saved chart: {chart_path}")
        else:
            print("\nNo products met the minimum negative review threshold.")
            print(f"Dataset has too few negative reviews for meaningful component analysis.")
    else:
        print("Run top_products_breakdown.py first")
