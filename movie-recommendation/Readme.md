# 🎬 Movie Recommendation System

A simple **content-based movie recommendation system** built using **Python** and **Streamlit**.  
The app suggests movies similar to the one selected by the user and displays their posters using the **TMDB API**.

---

## 🚀 Features

- Select a movie from a dropdown list
- Get **top 5 similar movie recommendations**
- Displays **movie posters** using TMDB API
- Automatically downloads required `.pkl` files from Google Drive at runtime
- Easy to deploy using **Streamlit Cloud**


## 🛠️ Tech Stack

- Python  
- Streamlit  
- Pandas  
- Pickle  
- TMDB API  
- gdown (for Google Drive file download)

## 📁 Project Structure

├── app.py
├── movies_dict.pkl
├── similarity.pkl
├── requirements.txt
└── README.md

> Note: The `.pkl` files are downloaded automatically when the app runs.


## ⚙️ How It Works

- Movie data is loaded from `movies_dict.pkl`
- Cosine similarity scores are loaded from `similarity.pkl`
- When a movie is selected:
  - Similarity scores are sorted
  - Top 5 similar movies are recommended
  - Movie posters are fetched using TMDB API

