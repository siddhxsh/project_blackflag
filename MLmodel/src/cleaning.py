import pandas as pd
import sys
import json
import os
import re

# Get workspace root
workspace_root = os.path.join(os.path.dirname(__file__), "..", "..")

# Load column mapping dynamically from column_analyzer.py output
mapping_file = os.path.join(workspace_root, "column_mapping.json")

if not os.path.exists(mapping_file):
    raise FileNotFoundError(f"{mapping_file} not found. Run column_analyzer.py first to generate the mapping.")

with open(mapping_file, 'r') as f:
    mapping_from_analyzer = json.load(f)

# Reverse the mapping: analyzer returns {StandardName: OriginalName}, but we need {OriginalName: StandardName}
COLUMN_MAPPING = {v: k for k, v in mapping_from_analyzer.items() if v is not None}

print(f"Column mapping loaded: {COLUMN_MAPPING}")

# Get input CSV filename from command line argument or use default
data_folder = os.path.join(workspace_root, "MLmodel", "data")

if len(sys.argv) > 1:
    input_filename = sys.argv[1]
else:
    # If no argument provided, look for any non-cleaned CSV file in data folder
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv') and 'cleaned' not in f.lower()]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {data_folder}. Usage: python cleaning.py <input_csv_filename>")
    input_filename = csv_files[0]

input_path = os.path.join(data_folder, input_filename)
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found: {input_path}")

output_filename = f"{os.path.splitext(input_filename)[0]}_cleaned.csv"
output_path = os.path.join(data_folder, output_filename)

print(f"Processing: {input_filename}")

df = pd.read_csv(input_path, encoding='latin-1')

# Rename columns based on dynamic mapping
df = df.rename(columns=COLUMN_MAPPING)

# Ensure required columns exist
required_cols = ["ProductName", "Price", "Rate", "Review", "Summary"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns after mapping: {missing_cols}")

# Keep only required columns, discard unnecessary ones
df = df[required_cols].copy()

# Normalize text columns so empty and whitespace-only entries are treated the same
for col in ["Review", "Summary"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

# Drop rows where both Review and Summary are empty
empty_mask = (df["Review"] == "") & (df["Summary"] == "")
df = df.loc[~empty_mask].copy()

# Remove exact duplicate rows
df = df.drop_duplicates()

# Validate Rate and keep only integers in {1,2,3,4,5}
df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
df = df[df["Rate"].isin([1, 2, 3, 4, 5])]
df["Rate"] = df["Rate"].astype(int)
if "Price" in df.columns:
    df["Price"] = df["Price"].astype(str).str.replace(r'[^\d]', '', regex=True)
    df["Price"] = pd.to_numeric(df["Price"], errors='coerce')

print("Cleaned shape:", df.shape)
print("Rate distribution:\n", df["Rate"].value_counts().sort_index())

# Create text column
df["text"] = df["Summary"] + " " + df["Review"]

# Emoji normalization uses emoji.demojize when available; falls back to no-op if package missing
try:
    import emoji  # type: ignore
except ImportError:
    emoji = None

url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
html_pattern = re.compile(r"<[^>]+>")
whitespace_pattern = re.compile(r"[\n\t\r]+")


def normalize_emojis(text: str) -> str:
    if emoji is None:
        return text
    # Convert emojis to :smile: style text, then strip punctuation-like colons/underscores
    demojized = emoji.demojize(text, language="en")
    demojized = demojized.replace(":", " ")
    demojized = re.sub(r"_+", " ", demojized)
    return demojized


def clean_text_value(text) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = html_pattern.sub(" ", text)
    text = url_pattern.sub(" ", text)
    text = whitespace_pattern.sub(" ", text)
    text = normalize_emojis(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Apply text cleaning to relevant columns
text_cols = [col for col in ["Summary", "Review", "text"] if col in df.columns]
for col in text_cols:
    df[col] = df[col].apply(clean_text_value)

# Map sentiment
def map_sentiment(r):
    if r >= 4: return "Positive"
    if r == 3: return "Neutral"
    return "Negative"

df["sentiment"] = df["Rate"].apply(map_sentiment)
df.to_csv(output_path, index=False)

print(f"\nCleaned data saved to: {output_path}")
