"""
Hybrid recommender: weighted blend of SVD (collaborative) + content-based scores.
Falls back to content-based for cold-start users.
"""

import numpy as np
import pandas as pd
from collaborative import SVDRecommender
from content_based import ContentBasedRecommender


class HybridRecommender:
    """
    Blends collaborative and content-based predictions.
    - Known users: alpha * SVD_score + (1-alpha) * content_score
    - Cold-start users (no history): pure content-based from seed movies
    """

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha  # weight for collaborative
        self.svd: SVDRecommender = None
        self.content: ContentBasedRecommender = None
        self.ratings: pd.DataFrame = None
        self.movies: pd.DataFrame = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.ratings = ratings
        self.movies = movies

        print("Training SVD (collaborative filtering)...")
        self.svd = SVDRecommender(n_factors=50).fit(ratings)

        print("Training content-based model...")
        self.content = ContentBasedRecommender().fit(movies)

        return self

    def recommend(self, user_id: int, n: int = 10) -> list[dict]:
        seen = set(self.ratings[self.ratings["user_id"] == user_id]["movie_id"].tolist())
        all_movies = set(self.movies["movie_id"].tolist())
        candidates = list(all_movies - seen)

        # Collaborative scores
        cf_scores = {}
        cf_recs = self.svd.recommend(user_id, n=len(candidates), seen_movies=seen)
        for r in cf_recs:
            cf_scores[r["movie_id"]] = r["predicted_rating"] / 5.0  # normalize to [0,1]

        # Content scores — based on top-rated seen movies
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        liked = user_ratings[user_ratings["rating"] >= 4]["movie_id"].tolist()
        cb_recs = self.content.recommend_for_user(liked, n=len(candidates), seen_movies=seen)
        cb_scores = {r["movie_id"]: r["score"] for r in cb_recs}

        # Blend
        blended = []
        for mid in candidates:
            cf = cf_scores.get(mid, self.svd.global_mean / 5.0)
            cb = cb_scores.get(mid, 0.0)
            score = self.alpha * cf + (1 - self.alpha) * cb
            blended.append({"movie_id": mid, "score": round(score, 4)})

        blended.sort(key=lambda x: x["score"], reverse=True)
        return blended[:n]

    def recommend_coldstart(self, seed_movie_ids: list[int], n: int = 10) -> list[dict]:
        return self.content.recommend_for_user(seed_movie_ids, n=n)

    def save(self, svd_path: str = None, content_path: str = None):
        self.svd.save(svd_path)
        self.content.save(content_path)

    @classmethod
    def load(cls, svd_path: str = None, content_path: str = None, alpha: float = 0.7) -> "HybridRecommender":
        h = cls(alpha=alpha)
        h.svd = SVDRecommender.load(svd_path)
        h.content = ContentBasedRecommender.load(content_path)
        return h
