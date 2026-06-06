"""
Content-based filtering using TF-IDF on movie titles + genre one-hot vectors.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

GENRE_COLS = [
    "Action", "Adventure", "Animation", "Childrens", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci_Fi", "Thriller", "War", "Western"
]


class ContentBasedRecommender:
    """
    Computes item-item cosine similarity from TF-IDF title + genre features.
    Recommends movies similar to what a user liked.
    """

    def __init__(self, tfidf_weight: float = 0.3, genre_weight: float = 0.7):
        self.tfidf_weight = tfidf_weight
        self.genre_weight = genre_weight
        self.similarity_matrix: np.ndarray = None
        self.movie_ids: list = []
        self.idx_to_movie: dict = {}
        self.movie_to_idx: dict = {}

    def fit(self, movies: pd.DataFrame):
        self.movie_ids = movies["movie_id"].tolist()
        self.movie_to_idx = {m: i for i, m in enumerate(self.movie_ids)}
        self.idx_to_movie = {i: m for m, i in self.movie_to_idx.items()}

        # TF-IDF on cleaned titles
        titles = movies["title"].fillna("").str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = tfidf.fit_transform(titles).toarray()

        # Genre vectors (already 0/1)
        genre_matrix = movies[GENRE_COLS].fillna(0).values.astype(float)
        genre_matrix = normalize(genre_matrix, norm="l2")

        # Weighted combination
        combined = np.hstack([
            self.tfidf_weight * tfidf_matrix,
            self.genre_weight * genre_matrix,
        ])
        self.similarity_matrix = cosine_similarity(combined)
        return self

    def similar_movies(self, movie_id: int, n: int = 10) -> list[dict]:
        if movie_id not in self.movie_to_idx:
            return []
        idx = self.movie_to_idx[movie_id]
        scores = self.similarity_matrix[idx]
        ranked = np.argsort(scores)[::-1]
        results = []
        for i in ranked:
            if self.idx_to_movie[i] != movie_id:
                results.append({"movie_id": self.idx_to_movie[i], "similarity": float(scores[i])})
            if len(results) >= n:
                break
        return results

    def recommend_for_user(
        self,
        liked_movie_ids: list[int],
        n: int = 10,
        seen_movies: set = None,
    ) -> list[dict]:
        if not liked_movie_ids:
            return []
        scores = np.zeros(len(self.movie_ids))
        for mid in liked_movie_ids:
            if mid in self.movie_to_idx:
                scores += self.similarity_matrix[self.movie_to_idx[mid]]
        scores /= len(liked_movie_ids)

        ranked = np.argsort(scores)[::-1]
        results = []
        for i in ranked:
            mid = self.idx_to_movie[i]
            if seen_movies and mid in seen_movies:
                continue
            if mid in liked_movie_ids:
                continue
            results.append({"movie_id": mid, "score": float(scores[i])})
            if len(results) >= n:
                break
        return results

    def save(self, path: str = None):
        path = path or os.path.join(MODEL_DIR, "content_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Content model saved to {path}")

    @classmethod
    def load(cls, path: str = None) -> "ContentBasedRecommender":
        path = path or os.path.join(MODEL_DIR, "content_model.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
