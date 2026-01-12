# Email & SMS Spam Detection (Email_spamML)

## Overview
This project detects whether a given SMS or email message is spam or not. It includes a notebook for model exploration and a Streamlit `app.py` for quick demos.

## How it works (high level)
- Data preprocessing: text cleaning, lowercasing, removing punctuation/stopwords, and optional lemmatization.
- Feature extraction: convert text to numerical features using `CountVectorizer` or `TfidfVectorizer` (TF–IDF).
- Model training: common choices are **Multinomial Naive Bayes** or **Logistic Regression** trained on the vectorized features.
- Evaluation: use train/test split and metrics like accuracy, precision, recall and F1 score.

## Why these choices
- TF–IDF highlights informative words while down-weighting common terms.
- Multinomial Naive Bayes is fast and works well for discrete word counts; Logistic Regression often gives stronger calibrated probabilities.

## Files
- `app.py`: Streamlit interface to input text and get a prediction.
- `sms-spam-detection.ipynb`: exploratory notebook with preprocessing, model training and evaluation.

## Dependencies
- Python 3.8+
- pandas, scikit-learn, streamlit, nltk (or spaCy)

Install with:

```bash
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

## Notes & improvements
- Try more advanced embeddings (word2vec, BERT) for better generalization.
- Use cross-validation and hyperparameter tuning for robust models.
- For production, add monitoring for concept drift and a safe feedback loop to re-train with new labeled data.
