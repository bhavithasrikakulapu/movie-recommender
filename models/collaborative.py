"""
Collaborative Filtering via SVD (matrix factorization).
Uses scikit-surprise or falls back to manual SVD via numpy.
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


class SVDRecommender:
    """
    Truncated SVD on user-item rating matrix.
    Predicts ratings as: R_hat = U * S * Vt + user_bias + item_bias + global_mean
    """

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_map: dict = {}
        self.item_map: dict = {}
        self.user_bias: np.ndarray = None
        self.item_bias: np.ndarray = None
        self.global_mean: float = 0.0
        self.U: np.ndarray = None
        self.S: np.ndarray = None
        self.Vt: np.ndarray = None
        self.predictions: np.ndarray = None

    def fit(self, ratings: pd.DataFrame):
        users = ratings["user_id"].unique()
        items = ratings["movie_id"].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {m: i for i, m in enumerate(items)}
        self.idx_to_item = {i: m for m, i in self.item_map.items()}

        n_users, n_items = len(users), len(items)
        self.global_mean = ratings["rating"].mean()

        # Build sparse matrix
        rows = ratings["user_id"].map(self.user_map).values
        cols = ratings["movie_id"].map(self.item_map).values
        vals = ratings["rating"].values.astype(float)

        mat = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
        mat_dense = mat.toarray()
        mask = mat_dense != 0

        # Compute biases only over observed entries (not zeros)
        user_sums = np.array(mat.sum(axis=1)).flatten()
        user_counts = np.diff(mat.tocsr().indptr)
        self.user_bias = np.where(
            user_counts > 0, user_sums / np.maximum(user_counts, 1) - self.global_mean, 0.0
        )

        item_sums = np.array(mat.sum(axis=0)).flatten()
        item_counts = np.diff(mat.tocsc().indptr)
        self.item_bias = np.where(
            item_counts > 0, item_sums / np.maximum(item_counts, 1) - self.global_mean, 0.0
        )

        # Demean matrix for SVD (only observed entries)
        mat_demeaned = np.where(
            mask,
            mat_dense - self.global_mean - self.user_bias[:, None] - self.item_bias[None, :],
            0,
        )

        k = min(self.n_factors, min(mat_demeaned.shape) - 1)
        self.U, self.S, self.Vt = svds(mat_demeaned, k=k)
        # Sort by singular values descending
        idx = np.argsort(self.S)[::-1]
        self.U, self.S, self.Vt = self.U[:, idx], self.S[idx], self.Vt[idx, :]

        self.predictions = (
            np.dot(self.U, np.dot(np.diag(self.S), self.Vt))
            + self.global_mean
            + self.user_bias[:, None]
            + self.item_bias[None, :]
        )
        self.predictions = np.clip(self.predictions, 1, 5)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user_map or movie_id not in self.item_map:
            return self.global_mean
        u = self.user_map[user_id]
        m = self.item_map[movie_id]
        return float(self.predictions[u, m])

    def recommend(self, user_id: int, n: int = 10, seen_movies: set = None) -> list[dict]:
        if user_id not in self.user_map:
            return []
        u = self.user_map[user_id]
        scores = self.predictions[u, :]
        item_scores = [(self.idx_to_item[i], float(scores[i])) for i in range(len(scores))]
        if seen_movies:
            item_scores = [(m, s) for m, s in item_scores if m not in seen_movies]
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return [{"movie_id": m, "predicted_rating": s} for m, s in item_scores[:n]]

    def save(self, path: str = None):
        path = path or os.path.join(MODEL_DIR, "svd_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"SVD model saved to {path}")

    @classmethod
    def load(cls, path: str = None) -> "SVDRecommender":
        path = path or os.path.join(MODEL_DIR, "svd_model.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
