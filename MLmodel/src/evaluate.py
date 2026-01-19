import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import find_cleaned_csv, get_model_dir

# ----------------------------
# CONFIG (MUST MATCH train.py)
# ----------------------------
TEXT_COLUMN = "text"        # SAME AS train.py
LABEL_COLUMN = "sentiment"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = get_model_dir()

# Auto-detect cleaned CSV
try:
    DATA_PATH = find_cleaned_csv(os.path.join(BASE_DIR, "data"))
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)

# CLEAN TEXT (SAME AS TRAINING)
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

# SAME SPLIT AS TRAINING (CRITICAL)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# LOAD SAVED MODEL & VECTORIZER
# ----------------------------
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.joblib"))

# ----------------------------
# TRANSFORM + PREDICT
# ----------------------------
X_test_tfidf = tfidf.transform(X_test)
y_pred = model.predict(X_test_tfidf)

# ----------------------------
# METRICS
# ----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ----------------------------
# ERROR ANALYSIS
# ----------------------------
results = pd.DataFrame({
    "text": X_test.values,
    "actual": y_test.values,
    "predicted": y_pred
})

errors = results[results["actual"] != results["predicted"]]

print("\nSample Misclassified Reviews:")
print(errors.sample(10))
