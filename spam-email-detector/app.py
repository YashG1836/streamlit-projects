import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()

import gdown
import os
VECTORIZER_PATH = "vectorizer.pkl"
MODEL_PATH = "model.pkl"

VECTORIZER_URL = "https://drive.google.com/uc?id=1z2b6sq9D3nf9oA_Bd1WDHwXtQRqhiQPv"
MODEL_URL = "https://drive.google.com/uc?id=1T9V81rlEmhVV9ONTbbZgZkuPH8lBDLjI"

if not os.path.exists(VECTORIZER_PATH):
    gdown.download(VECTORIZER_URL, VECTORIZER_PATH, quiet=False)

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

with open(VECTORIZER_PATH, "rb") as f:
    tfidf = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
    
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")