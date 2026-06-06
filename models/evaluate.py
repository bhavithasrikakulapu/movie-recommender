"""
Offline evaluation: RMSE, MAE, Precision@K, Recall@K, NDCG@K.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from collaborative import SVDRecommender
from content_based import ContentBasedRecommender

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    hits = sum(1 for m in top_k if m in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    hits = sum(1 for m in top_k if m in relevant)
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(
        (1.0 / np.log2(i + 2)) for i, m in enumerate(top_k) if m in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_svd(ratings: pd.DataFrame, movies: pd.DataFrame, n_factors: int = 50, k: int = 10):
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    model = SVDRecommender(n_factors=n_factors).fit(train)

    # Rating prediction metrics
    y_true, y_pred = [], []
    for _, row in test.iterrows():
        pred = model.predict(int(row["user_id"]), int(row["movie_id"]))
        y_true.append(row["rating"])
        y_pred.append(pred)

    rmse_val = rmse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)

    # Ranking metrics
    test_users = test["user_id"].unique()[:200]  # sample for speed
    precisions, recalls, ndcgs = [], [], []

    for uid in test_users:
        relevant = set(test[(test["user_id"] == uid) & (test["rating"] >= 4)]["movie_id"])
        if not relevant:
            continue
        seen = set(train[train["user_id"] == uid]["movie_id"])
        recs = model.recommend(uid, n=k * 5, seen_movies=seen)
        rec_ids = [r["movie_id"] for r in recs]
        precisions.append(precision_at_k(rec_ids, relevant, k))
        recalls.append(recall_at_k(rec_ids, relevant, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k))

    results = {
        "model": "SVD Collaborative Filtering",
        "n_factors": n_factors,
        "RMSE": round(rmse_val, 4),
        "MAE": round(mae_val, 4),
        f"Precision@{k}": round(np.mean(precisions), 4),
        f"Recall@{k}": round(np.mean(recalls), 4),
        f"NDCG@{k}": round(np.mean(ndcgs), 4),
    }
    return results, model, train


def run_full_evaluation(ratings: pd.DataFrame, movies: pd.DataFrame):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n=== Evaluating SVD Model ===")
    results, model, train = evaluate_svd(ratings, movies)
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Save results
    pd.DataFrame([results]).to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/metrics.csv")
    return results, model, train
