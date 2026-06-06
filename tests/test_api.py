"""
API integration tests using FastAPI TestClient.
Requires trained models in models/ directory.
Run: pytest tests/test_api.py -v
"""

import sys
import os
import pytest

# Skip if models not trained yet
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "models", "svd_model.pkl")),
    reason="Models not trained. Run: python models/train.py first."
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "request_counts" in data
    assert "avg_latency_ms" in data


def test_recommend_known_user(client):
    r = client.post("/recommend", json={"user_id": 1, "n": 5})
    assert r.status_code == 200
    recs = r.json()
    assert len(recs) <= 5
    assert all("movie_id" in rec and "title" in rec and "score" in rec for rec in recs)


def test_recommend_unknown_user(client):
    r = client.post("/recommend", json={"user_id": 99999, "n": 5})
    assert r.status_code == 404


def test_recommend_coldstart(client):
    r = client.post("/recommend/coldstart", json={"seed_movie_ids": [1, 50, 258], "n": 5})
    assert r.status_code == 200
    recs = r.json()
    assert len(recs) <= 5


def test_similar_movies(client):
    r = client.get("/movies/similar/1?n=5")
    assert r.status_code == 200
    recs = r.json()
    assert all(rec["movie_id"] != 1 for rec in recs)


def test_get_movie(client):
    r = client.get("/movies/1")
    assert r.status_code == 200
    assert "title" in r.json()


def test_get_unknown_movie(client):
    r = client.get("/movies/999999")
    assert r.status_code == 404
