"""
Project 4: Sentiment Analysis on Text Data


Goal: Classify movie reviews as positive or negative sentiment using
text preprocessing, TF-IDF feature extraction, and classification.
"""

# ---------- STEP 0: Import Libraries ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style("darkgrid")

# Download required NLTK data (only downloads once, cached after that)
nltk.download("movie_reviews", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ---------- STEP 1: Data Collection ----------
# NLTK's movie_reviews corpus: 2000 real movie reviews (1000 positive,
# 1000 negative) - same style of data as the IMDB dataset suggested
# in the project brief, built directly into NLTK.
reviews = []
labels = []
for fileid in movie_reviews.fileids():
    category = movie_reviews.categories(fileid)[0]  # 'pos' or 'neg'
    text = movie_reviews.raw(fileid)
    reviews.append(text)
    labels.append(1 if category == "pos" else 0)  # 1 = positive, 0 = negative

df = pd.DataFrame({"review": reviews, "sentiment": labels})
print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df["sentiment"].value_counts())
print("\nSample review (first 300 chars):\n", df["review"].iloc[0][:300])

# ---------- STEP 2: Text Preprocessing ----------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Cleans and normalizes a single review's text."""
    text = text.lower()                                   # lowercase everything
    text = re.sub(r"[^a-z\s]", "", text)                    # remove numbers/punctuation
    tokens = text.split()                                   # tokenization (split into words)
    tokens = [w for w in tokens if w not in stop_words]      # remove stop words (the, is, a...)
    tokens = [lemmatizer.lemmatize(w) for w in tokens]        # lemmatize (running -> run)
    tokens = [w for w in tokens if len(w) > 2]                # drop very short tokens
    return " ".join(tokens)

print("\nCleaning text (this takes a moment for 2000 reviews)...")
df["cleaned_review"] = df["review"].apply(clean_text)
print("Done.")
print("\nBefore cleaning:\n", df["review"].iloc[0][:200])
print("\nAfter cleaning:\n", df["cleaned_review"].iloc[0][:200])

# ---------- STEP 3: Feature Extraction (TF-IDF) ----------
# TF-IDF converts text into numeric vectors: it weighs words by how
# important/unique they are to a review, not just how often they appear.
X = df["cleaned_review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)  # keep top 5000 words only
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"\nTF-IDF matrix shape (train): {X_train_tfidf.shape}")

# ---------- STEP 4: Model Building & Training ----------
# Model 1: Multinomial Naive Bayes (classic, fast, works well for text)
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Model 2: Logistic Regression (often stronger baseline for text too)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# ---------- STEP 5: Predictions & Evaluation ----------
nb_pred = nb_model.predict(X_test_tfidf)
lr_pred = lr_model.predict(X_test_tfidf)

nb_acc = accuracy_score(y_test, nb_pred)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"\nNaive Bayes Accuracy: {nb_acc:.4f}")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

print("\nClassification Report (Logistic Regression):\n",
      classification_report(y_test, lr_pred, target_names=["Negative", "Positive"]))

# ---------- STEP 6: Visualization ----------

# 6a. Class distribution
plt.figure(figsize=(6, 5))
sns.countplot(x=df["sentiment"].map({0: "Negative", 1: "Positive"}),
              hue=df["sentiment"].map({0: "Negative", 1: "Positive"}),
              palette=["#E74C3C", "#2ECC71"], legend=False)
plt.title("Sentiment Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.close()

# 6b. Confusion matrix for Logistic Regression
cm = confusion_matrix(y_test, lr_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="mako",
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix — Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# 6c. Model comparison
plt.figure(figsize=(6, 5))
plt.bar(["Naive Bayes", "Logistic Regression"], [nb_acc, lr_acc],
        color=["#4C9AFF", "#2ECC71"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# 6d. Top positive/negative words (from Logistic Regression coefficients)
feature_names = np.array(vectorizer.get_feature_names_out())
coefs = lr_model.coef_[0]
top_positive_idx = np.argsort(coefs)[-15:]
top_negative_idx = np.argsort(coefs)[:15]

plt.figure(figsize=(9, 6))
words = list(feature_names[top_negative_idx]) + list(feature_names[top_positive_idx])
scores = list(coefs[top_negative_idx]) + list(coefs[top_positive_idx])
colors = ["#E74C3C"] * 15 + ["#2ECC71"] * 15
plt.barh(words, scores, color=colors)
plt.title("Top Words Driving Negative (red) vs Positive (green) Sentiment")
plt.xlabel("Coefficient (impact on sentiment)")
plt.tight_layout()
plt.savefig("top_sentiment_words.png", dpi=150)
plt.close()

print("\nAll plots saved successfully.")

# ---------- STEP 7: Key Insights ----------
print("\nKey Insights:")
print(f"1. Logistic Regression ({lr_acc*100:.1f}%) outperformed Naive Bayes "
      f"({nb_acc*100:.1f}%) on this dataset.")
print(f"2. Strongest positive word: '{feature_names[top_positive_idx[-1]]}'")
print(f"3. Strongest negative word: '{feature_names[top_negative_idx[0]]}'")
print("4. TF-IDF with 5000 features was enough to capture meaningful patterns "
      "in review text without needing deep learning.")