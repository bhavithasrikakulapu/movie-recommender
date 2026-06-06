"""
Unit tests for collaborative, content-based, and API endpoints.
Run: pytest tests/ -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from collaborative import SVDRecommender
from content_based import ContentBasedRecommender
from evaluate import rmse, mae, precision_at_k, recall_at_k, ndcg_at_k


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ratings():
    data = [
        (1, 1, 5), (1, 2, 3), (1, 3, 4), (1, 4, 1), (1, 5, 2),
        (2, 1, 4), (2, 2, 5), (2, 3, 3), (2, 4, 2), (2, 5, 4),
        (3, 1, 1), (3, 2, 2), (3, 3, 5), (3, 4, 4), (3, 5, 3),
        (4, 1, 3), (4, 2, 4), (4, 3, 2), (4, 4, 5), (4, 5, 1),
        (5, 1, 2), (5, 2, 1), (5, 3, 4), (5, 4, 3), (5, 5, 5),
    ]
    return pd.DataFrame(data, columns=["user_id", "movie_id", "rating"])


@pytest.fixture
def sample_movies():
    rows = []
    genres = ["Action", "Adventure", "Animation", "Childrens", "Comedy",
              "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir",
              "Horror", "Musical", "Mystery", "Romance", "Sci_Fi",
              "Thriller", "War", "Western"]
    for i in range(1, 6):
        row = {"movie_id": i, "title": f"Movie {i} (200{i})", "year": 2001 + i}
        for j, g in enumerate(genres):
            row[g] = 1 if j % (i + 1) == 0 else 0
        rows.append(row)
    return pd.DataFrame(rows)


# ── SVD tests ─────────────────────────────────────────────────────────────────

class TestSVDRecommender:
    def test_fit_returns_self(self, sample_ratings):
        model = SVDRecommender(n_factors=2)
        result = model.fit(sample_ratings)
        assert result is model

    def test_predict_in_range(self, sample_ratings):
        model = SVDRecommender(n_factors=2).fit(sample_ratings)
        for uid in [1, 2, 3]:
            for mid in [1, 2, 3]:
                pred = model.predict(uid, mid)
                assert 1.0 <= pred <= 5.0, f"Pred {pred} out of [1,5]"

    def test_predict_unknown_user(self, sample_ratings):
        model = SVDRecommender(n_factors=2).fit(sample_ratings)
        pred = model.predict(999, 1)
        assert pred == model.global_mean

    def test_recommend_excludes_seen(self, sample_ratings):
        model = SVDRecommender(n_factors=2).fit(sample_ratings)
        seen = {1, 2}
        recs = model.recommend(1, n=5, seen_movies=seen)
        rec_ids = {r["movie_id"] for r in recs}
        assert not rec_ids.intersection(seen)

    def test_recommend_returns_n(self, sample_ratings):
        model = SVDRecommender(n_factors=2).fit(sample_ratings)
        recs = model.recommend(1, n=3, seen_movies={1, 2, 3})
        assert len(recs) <= 3


# ── Content-based tests ───────────────────────────────────────────────────────

class TestContentBasedRecommender:
    def test_fit(self, sample_movies):
        model = ContentBasedRecommender().fit(sample_movies)
        assert model.similarity_matrix is not None
        assert model.similarity_matrix.shape == (5, 5)

    def test_similar_movies(self, sample_movies):
        model = ContentBasedRecommender().fit(sample_movies)
        recs = model.similar_movies(1, n=3)
        assert len(recs) <= 3
        assert all("movie_id" in r and "similarity" in r for r in recs)
        assert all(r["movie_id"] != 1 for r in recs)

    def test_recommend_for_user(self, sample_movies):
        model = ContentBasedRecommender().fit(sample_movies)
        recs = model.recommend_for_user([1, 2], n=3, seen_movies={1, 2})
        assert all(r["movie_id"] not in {1, 2} for r in recs)

    def test_unknown_movie(self, sample_movies):
        model = ContentBasedRecommender().fit(sample_movies)
        recs = model.similar_movies(999, n=5)
        assert recs == []


# ── Metric tests ──────────────────────────────────────────────────────────────

class TestMetrics:
    def test_rmse_perfect(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0

    def test_mae_perfect(self):
        assert mae([1, 2, 3], [1, 2, 3]) == 0.0

    def test_rmse_known(self):
        result = rmse([4], [3])
        assert abs(result - 1.0) < 1e-6

    def test_precision_at_k(self):
        recs = [1, 2, 3, 4, 5]
        relevant = {1, 3, 6}
        assert precision_at_k(recs, relevant, k=5) == pytest.approx(2 / 5)

    def test_recall_at_k(self):
        recs = [1, 2, 3, 4, 5]
        relevant = {1, 3, 6}
        assert recall_at_k(recs, relevant, k=5) == pytest.approx(2 / 3)

    def test_ndcg_at_k_perfect(self):
        recs = [1, 2, 3]
        relevant = {1, 2, 3}
        assert ndcg_at_k(recs, relevant, k=3) == pytest.approx(1.0)

    def test_ndcg_empty_relevant(self):
        assert ndcg_at_k([1, 2], set(), k=2) == 0.0
