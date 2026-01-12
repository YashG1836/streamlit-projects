# import streamlit as st
# import pickle
# import pandas as pd


# with open("movies_dict.pkl", "rb") as f:
#     dataset = pickle.load(f)
# with open("similarity.pkl", "rb") as f:
#     similarity = pickle.load(f)

# movies = pd.DataFrame(dataset)
# movies_list = movies["title"].values

# def recommend(movie):
#     movie_index = movies[movies["title"] == movie].index[0]
#     similar = sorted(
#         list(enumerate(similarity[movie_index])),
#         reverse=True,
#         key=lambda x: x[1],
#     )[1:6]
#     return [movies.iloc[i[0]].title for i in similar]

# st.title("Movie Recommender system")
# option = st.selectbox("Search movies here", movies_list)

# if st.button("Recommend"):
#     for title in recommend(option):
#         st.write(title)

import streamlit as st
import pickle
import pandas as pd
import requests
import os
import gdown
MOVIES_PATH = "movies_dict.pkl"
SIMILARITY_PATH = "similarity.pkl"

MOVIES_URL = "https://drive.google.com/uc?id=10S6RarYpH6BgG0gylcSfX9oW5Sbgskvr"
SIMILARITY_URL = "https://drive.google.com/uc?id=126H77HChKoJR83Zo415dMLihj7ojWuhP"

if not os.path.exists(MOVIES_PATH):
    gdown.download(MOVIES_URL, MOVIES_PATH, quiet=False)

if not os.path.exists(SIMILARITY_PATH):
    gdown.download(SIMILARITY_URL, SIMILARITY_PATH, quiet=False)

with open(MOVIES_PATH, "rb") as f:
    dataset = pickle.load(f)

with open(SIMILARITY_PATH, "rb") as f:
    similarity = pickle.load(f)
movies = pd.DataFrame(dataset)
movies_list = movies["title"].values

def get_tmdb_api_key() -> str:
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return os.environ.get("TMDB_API_KEY", "")

TMDB_API_KEY = get_tmdb_api_key()

def fetch_poster(title: str) -> str:
    if not TMDB_API_KEY:
        return ""
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"query": title}
        headers = {}
        # TMDB v4 tokens start with 'eyJ'; v3 keys use the api_key param.
        if TMDB_API_KEY.startswith("eyJ"):
            headers["Authorization"] = f"Bearer {TMDB_API_KEY}"
        else:
            params["api_key"] = TMDB_API_KEY

        resp = requests.get(url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return ""
        poster_path = results[0].get("poster_path")
        if not poster_path:
            return ""
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        return ""

def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    similar = sorted(
        list(enumerate(similarity[movie_index])),
        reverse=True,
        key=lambda x: x[1],
    )[1:6]
    out = []
    for idx, _ in similar:
        title = movies.iloc[idx].title
        poster = fetch_poster(title)
        out.append((title, poster))
    return out

st.title("Movie Recommender system")
option = st.selectbox("Search movies here", movies_list)

if st.button("Recommend"):
    cols = st.columns(5)
    for col, (title, poster) in zip(cols, recommend(option)):
        with col:
            if poster:
                st.image(poster, use_container_width=True)
            st.caption(title)