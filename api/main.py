"""
FastAPI app — Movie Recommendation System
Endpoints:
  GET  /health
  GET  /metrics
  POST /recommend          (known user)
  POST /recommend/coldstart (seed movies → recs)
  GET  /movies/similar/{movie_id}
  GET  /movies/{movie_id}
"""

import os
import sys
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Allow imports from models/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from collaborative import SVDRecommender
from content_based import ContentBasedRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Global state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    t0 = time.time()
    try:
        app_state["svd"] = SVDRecommender.load(os.path.join(MODEL_DIR, "svd_model.pkl"))
        logger.info("SVD model loaded")
        app_state["content"] = ContentBasedRecommender.load(os.path.join(MODEL_DIR, "content_model.pkl"))
        logger.info("Content model loaded")
    except Exception:
        logger.exception("Failed to load models")
        raise
    app_state["ratings"] = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    app_state["movies"] = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))

    # Build movie lookup
    movies_df = app_state["movies"]
    app_state["movie_lookup"] = movies_df.set_index("movie_id").to_dict("index")

    # Load metrics if available
    metrics_path = os.path.join(BASE_DIR, "results", "metadata.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            app_state["model_metrics"] = json.load(f)
    else:
        app_state["model_metrics"] = {}

    # Request counter for monitoring
    app_state["request_counts"] = {}
    app_state["latencies"] = []
    app_state["startup_time"] = time.time() - t0

    logger.info(f"Models loaded in {app_state['startup_time']:.2f}s")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Movie Recommendation API",
    description="End-to-end ML movie recommendation system (SVD + content-based hybrid)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    latency = time.time() - t0
    endpoint = request.url.path
    counts = app_state.get("request_counts", {})
    counts[endpoint] = counts.get(endpoint, 0) + 1
    app_state["request_counts"] = counts
    latencies = app_state.get("latencies", [])
    latencies.append(latency)
    if len(latencies) > 1000:
        latencies = latencies[-1000:]
    app_state["latencies"] = latencies
    return response


# ── Request/Response schemas ──────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: int = Field(..., json_schema_extra={"example": 1})
    n: int = Field(10, ge=1, le=50)
    alpha: float = Field(0.7, ge=0.0, le=1.0, description="Weight for collaborative score")


class ColdStartRequest(BaseModel):
    seed_movie_ids: list[int] = Field(..., json_schema_extra={"example": [1, 50, 258]})
    n: int = Field(10, ge=1, le=50)


class MovieRec(BaseModel):
    movie_id: int
    title: str
    year: Optional[float]
    genres: list[str]
    score: float


# ── Helpers ───────────────────────────────────────────────────────────────────

GENRE_COLS = [
    "Action", "Adventure", "Animation", "Childrens", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci_Fi", "Thriller", "War", "Western"
]


def enrich(movie_id: int, score: float) -> dict:
    lookup = app_state["movie_lookup"]
    info = lookup.get(movie_id, {})
    genres = [g for g in GENRE_COLS if info.get(g, 0) == 1]
    return {
        "movie_id": movie_id,
        "title": info.get("title", "Unknown"),
        "year": info.get("year"),
        "genres": genres,
        "score": round(score, 4),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "startup_time_s": round(app_state.get("startup_time", 0), 2),
        "models": ["svd", "content_based"],
    }


@app.get("/metrics")
def metrics():
    latencies = app_state.get("latencies", [])
    import numpy as np
    return {
        "request_counts": app_state.get("request_counts", {}),
        "avg_latency_ms": round(np.mean(latencies) * 1000, 2) if latencies else 0,
        "p95_latency_ms": round(np.percentile(latencies, 95) * 1000, 2) if latencies else 0,
        "model_metrics": app_state.get("model_metrics", {}),
    }


@app.post("/recommend", response_model=list[MovieRec])
def recommend(req: RecommendRequest):
    svd: SVDRecommender = app_state["svd"]
    content: ContentBasedRecommender = app_state["content"]
    ratings: pd.DataFrame = app_state["ratings"]

    if req.user_id not in svd.user_map:
        raise HTTPException(status_code=404, detail=f"User {req.user_id} not found. Use /recommend/coldstart.")

    seen = set(ratings[ratings["user_id"] == req.user_id]["movie_id"].tolist())
    all_movies = set(app_state["movies"]["movie_id"].tolist())
    candidates = list(all_movies - seen)

    # SVD scores
    cf_recs = svd.recommend(req.user_id, n=len(candidates), seen_movies=seen)
    cf_scores = {r["movie_id"]: r["predicted_rating"] / 5.0 for r in cf_recs}

    # Content scores
    user_ratings = ratings[ratings["user_id"] == req.user_id]
    liked = user_ratings[user_ratings["rating"] >= 4]["movie_id"].tolist()
    cb_recs = content.recommend_for_user(liked, n=len(candidates), seen_movies=seen)
    cb_scores = {r["movie_id"]: r["score"] for r in cb_recs}

    blended = []
    for mid in candidates:
        cf = cf_scores.get(mid, svd.global_mean / 5.0)
        cb = cb_scores.get(mid, 0.0)
        score = req.alpha * cf + (1 - req.alpha) * cb
        blended.append((mid, score))

    blended.sort(key=lambda x: x[1], reverse=True)
    return [enrich(mid, score) for mid, score in blended[: req.n]]


@app.post("/recommend/coldstart", response_model=list[MovieRec])
def recommend_coldstart(req: ColdStartRequest):
    content: ContentBasedRecommender = app_state["content"]
    recs = content.recommend_for_user(req.seed_movie_ids, n=req.n, seen_movies=set(req.seed_movie_ids))
    if not recs:
        raise HTTPException(status_code=404, detail="No recommendations found for given seed movies.")
    return [enrich(r["movie_id"], r["score"]) for r in recs]


@app.get("/movies/similar/{movie_id}", response_model=list[MovieRec])
def similar_movies(movie_id: int, n: int = 10):
    content: ContentBasedRecommender = app_state["content"]
    recs = content.similar_movies(movie_id, n=n)
    if not recs:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found.")
    return [enrich(r["movie_id"], r["similarity"]) for r in recs]


@app.get("/movies/{movie_id}")
def get_movie(movie_id: int):
    lookup = app_state["movie_lookup"]
    if movie_id not in lookup:
        raise HTTPException(status_code=404, detail="Movie not found.")
    return enrich(movie_id, 0.0)
