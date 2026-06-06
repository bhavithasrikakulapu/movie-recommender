"""
End-to-end training pipeline.
Run: python models/train.py
"""

import os
import sys
import json
import time
import pandas as pd

# Allow imports from data/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from collaborative import SVDRecommender
from content_based import ContentBasedRecommender
from evaluate import run_full_evaluation

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading processed data...")
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    print(f"  Ratings: {ratings.shape}, Movies: {movies.shape}")

    # Evaluate and get trained SVD on full train split
    print("\n--- Running evaluation ---")
    metrics, svd_model, train_ratings = run_full_evaluation(ratings, movies)

    # Train final models on ALL data
    print("\n--- Training final models on full dataset ---")
    t0 = time.time()
    final_svd = SVDRecommender(n_factors=50).fit(ratings)
    print(f"  SVD trained in {time.time()-t0:.1f}s")

    t0 = time.time()
    content_model = ContentBasedRecommender().fit(movies)
    print(f"  Content model trained in {time.time()-t0:.1f}s")

    # Save
    final_svd.save(os.path.join(MODEL_DIR, "svd_model.pkl"))
    content_model.save(os.path.join(MODEL_DIR, "content_model.pkl"))

    # Save metrics + metadata
    metadata = {
        "metrics": metrics,
        "num_users": int(ratings["user_id"].nunique()),
        "num_movies": int(ratings["movie_id"].nunique()),
        "num_ratings": int(len(ratings)),
        "svd_factors": 50,
    }
    with open(os.path.join(RESULTS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Training Complete ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
