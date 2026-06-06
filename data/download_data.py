"""
Download MovieLens 100K dataset (free, no auth required).
Run: python data/download_data.py
"""

import os
import urllib.request
import zipfile
import pandas as pd

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, "raw")


def download():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "ml-100k.zip")
    if not os.path.exists(zip_path):
        print("Downloading MovieLens 100K...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        print("Done.")
    else:
        print("Already downloaded.")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)
    print(f"Extracted to {RAW_DIR}")


def load_ratings() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "ml-100k", "u.data")
    df = pd.read_csv(path, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
    return df


def load_movies() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "ml-100k", "u.item")
    cols = [
        "movie_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci_Fi", "Thriller", "War", "Western"
    ]
    df = pd.read_csv(path, sep="|", names=cols, encoding="latin-1")
    return df


if __name__ == "__main__":
    download()
    ratings = load_ratings()
    movies = load_movies()
    print(f"Ratings: {ratings.shape}, Movies: {movies.shape}")
    print(ratings.head())
    print(movies[["movie_id", "title"]].head())
