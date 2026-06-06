"""
Data cleaning and feature engineering for MovieLens 100K.
"""

import os
import pandas as pd
import numpy as np
from download_data import load_ratings, load_movies

GENRE_COLS = [
    "Action", "Adventure", "Animation", "Childrens", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci_Fi", "Thriller", "War", "Western"
]

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    df = ratings.copy()
    # Drop duplicates (keep last rating per user-movie pair)
    df = df.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
    # Remove users with < 5 ratings and movies with < 5 ratings
    user_counts = df["user_id"].value_counts()
    movie_counts = df["movie_id"].value_counts()
    df = df[df["user_id"].isin(user_counts[user_counts >= 5].index)]
    df = df[df["movie_id"].isin(movie_counts[movie_counts >= 5].index)]
    df = df.reset_index(drop=True)
    return df


def engineer_movie_features(movies: pd.DataFrame) -> pd.DataFrame:
    df = movies.copy()
    # Parse release year
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype(float)
    df["year"] = df["year"].fillna(df["year"].median())
    # Genre vector already one-hot — fill NaN with 0
    df[GENRE_COLS] = df[GENRE_COLS].fillna(0)
    # Popularity proxy: will be merged from ratings
    return df[["movie_id", "title", "year"] + GENRE_COLS]


def build_user_features(ratings: pd.DataFrame) -> pd.DataFrame:
    """User-level aggregate features for cold-start handling."""
    uf = ratings.groupby("user_id").agg(
        num_ratings=("rating", "count"),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
    ).reset_index()
    uf["std_rating"] = uf["std_rating"].fillna(0)
    return uf


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    ratings = load_ratings()
    movies = load_movies()

    ratings_clean = clean_ratings(ratings)
    movie_features = engineer_movie_features(movies)
    user_features = build_user_features(ratings_clean)

    # Add movie popularity
    popularity = ratings_clean.groupby("movie_id").size().rename("num_ratings_movie")
    movie_features = movie_features.merge(popularity, on="movie_id", how="left").fillna(0)

    ratings_clean.to_csv(os.path.join(PROCESSED_DIR, "ratings.csv"), index=False)
    movie_features.to_csv(os.path.join(PROCESSED_DIR, "movies.csv"), index=False)
    user_features.to_csv(os.path.join(PROCESSED_DIR, "users.csv"), index=False)

    print(f"Cleaned ratings: {ratings_clean.shape}")
    print(f"Movie features: {movie_features.shape}")
    print(f"User features: {user_features.shape}")

    # Basic stats
    print("\n--- Data Quality Report ---")
    print(f"Rating range: {ratings_clean['rating'].min()} - {ratings_clean['rating'].max()}")
    print(f"Unique users: {ratings_clean['user_id'].nunique()}")
    print(f"Unique movies: {ratings_clean['movie_id'].nunique()}")
    sparsity = 1 - len(ratings_clean) / (ratings_clean["user_id"].nunique() * ratings_clean["movie_id"].nunique())
    print(f"Matrix sparsity: {sparsity:.4f}")

    return ratings_clean, movie_features, user_features


if __name__ == "__main__":
    run()
