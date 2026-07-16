# 🎬 Movie Review Sentiment Analysis

Built a sentiment analysis pipeline to classify movie reviews as positive or negative using NLP text preprocessing, TF-IDF feature extraction, and classification models. Part of Jyesta Data Science Internship.

## 📌 Overview
This project covers the full NLP workflow — text cleaning, tokenization, lemmatization, TF-IDF vectorization, model training, and evaluation — using NLTK's `movie_reviews` corpus (2000 real reviews, same style as IMDB data).

## 🛠️ Tools & Libraries
- Python
- NLTK (text preprocessing)
- Scikit-learn (TF-IDF, models)
- Pandas, NumPy, Matplotlib, Seaborn

## 🔍 Steps Followed
1. **Data Collection** — Loaded 2000 labeled movie reviews (1000 positive, 1000 negative) from NLTK.
2. **Text Preprocessing** — Lowercasing, punctuation/number removal, stop word removal, and lemmatization.
3. **Feature Extraction** — Converted cleaned text into numeric vectors using TF-IDF (top 5000 features).
4. **Model Building** — Trained Naive Bayes and Logistic Regression classifiers.
5. **Model Evaluation** — Compared accuracy, precision, recall, and F1-score.
6. **Word Analysis** — Identified the most influential positive/negative words.

## 📊 Results

| Model               | Accuracy |
|---------------------|----------|
| Naive Bayes          | 80.75%   |
| Logistic Regression  | 83.25%   |

**Top positive word:** `great` | **Top negative word:** `bad`

## 📁 Files
- `sentiment_analysis.py` — full commented Python script
- `*.png` — class distribution, confusion matrix, model comparison, top words
