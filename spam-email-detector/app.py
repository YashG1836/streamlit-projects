import os
import string
import pickle
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gdown

# -------------------- NLTK SETUP (CRITICAL FIX) --------------------
@st.cache_resource
def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

setup_nltk()

ps = PorterStemmer()

# -------------------- MODEL FILES --------------------
VECTORIZER_PATH = "vectorizer.pkl"
MODEL_PATH = "model.pkl"

VECTORIZER_URL = "https://drive.google.com/uc?id=1z2b6sq9D3nf9oA_Bd1WDHwXtQRqhiQPv"
MODEL_URL = "https://drive.google.com/uc?id=1T9V81rlEmhVV9ONTbbZgZkuPH8lBDLjI"

# -------------------- DOWNLOAD MODELS --------------------
if not os.path.exists(VECTORIZER_PATH):
    gdown.download(VECTORIZER_URL, VECTORIZER_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

with open(VECTORIZER_PATH, "rb") as f:
    tfidf = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -------------------- TEXT PREPROCESSING --------------------
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tokens = [i for i in tokens if i.isalnum()]
    tokens = [
        i for i in tokens
        if i not in stopwords.words("english")
        and i not in string.punctuation
    ]

    tokens = [ps.stem(i) for i in tokens]
    return " ".join(tokens)

# -------------------- STREAMLIT UI --------------------
st.title("Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if not input_sms.strip():
        st.warning("Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
