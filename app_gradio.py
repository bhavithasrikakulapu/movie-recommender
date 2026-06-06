"""
Gradio demo for Hugging Face Spaces.
Provides a UI to get movie recommendations.
Deploy: push this file + models/ + data/ to HF Space (Docker SDK).
Run locally: python app_gradio.py
"""

import os
import sys
import pandas as pd
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from collaborative import SVDRecommender
from content_based import ContentBasedRecommender

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

# Load models once
print("Loading models...")
svd = SVDRecommender.load(os.path.join(MODEL_DIR, "svd_model.pkl"))
content = ContentBasedRecommender.load(os.path.join(MODEL_DIR, "content_model.pkl"))
ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
movie_lookup = movies.set_index("movie_id").to_dict("index")

GENRE_COLS = [
    "Action", "Adventure", "Animation", "Childrens", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci_Fi", "Thriller", "War", "Western"
]

ALL_MOVIE_TITLES = sorted(movies["title"].dropna().tolist())
TITLE_TO_ID = dict(zip(movies["title"], movies["movie_id"]))


def format_rec(movie_id: int, score: float) -> str:
    info = movie_lookup.get(movie_id, {})
    title = info.get("title", f"Movie {movie_id}")
    genres = ", ".join(g for g in GENRE_COLS if info.get(g, 0) == 1) or "Unknown"
    year = info.get("year", "")
    year_str = f" ({int(year)})" if year else ""
    return f"**{title}{year_str}** · Genres: {genres} · Score: {score:.3f}"


def recommend_for_user(user_id: int, n: int, alpha: float):
    if user_id not in svd.user_map:
        return f"❌ User {user_id} not found in training data (valid range: 1–943)."

    seen = set(ratings[ratings["user_id"] == user_id]["movie_id"].tolist())
    all_movies = set(movies["movie_id"].tolist())
    candidates = list(all_movies - seen)

    cf_recs = svd.recommend(user_id, n=len(candidates), seen_movies=seen)
    cf_scores = {r["movie_id"]: r["predicted_rating"] / 5.0 for r in cf_recs}

    user_ratings = ratings[ratings["user_id"] == user_id]
    liked = user_ratings[user_ratings["rating"] >= 4]["movie_id"].tolist()
    cb_recs = content.recommend_for_user(liked, n=len(candidates), seen_movies=seen)
    cb_scores = {r["movie_id"]: r["score"] for r in cb_recs}

    blended = []
    for mid in candidates:
        cf = cf_scores.get(mid, svd.global_mean / 5.0)
        cb = cb_scores.get(mid, 0.0)
        blended.append((mid, alpha * cf + (1 - alpha) * cb))

    blended.sort(key=lambda x: x[1], reverse=True)
    output = [f"### Recommendations for User {user_id}\n"]
    output.append(f"*User has rated {len(seen)} movies · Hybrid: {alpha:.0%} collaborative + {1-alpha:.0%} content*\n")
    for i, (mid, score) in enumerate(blended[:n], 1):
        output.append(f"{i}. {format_rec(mid, score)}")
    return "\n".join(output)


def recommend_coldstart(selected_titles: list, n: int):
    if not selected_titles:
        return "❌ Please select at least one movie."
    seed_ids = [TITLE_TO_ID[t] for t in selected_titles if t in TITLE_TO_ID]
    recs = content.recommend_for_user(seed_ids, n=n, seen_movies=set(seed_ids))
    if not recs:
        return "No recommendations found."
    output = [f"### Because you liked: {', '.join(selected_titles)}\n"]
    for i, r in enumerate(recs, 1):
        output.append(f"{i}. {format_rec(r['movie_id'], r['score'])}")
    return "\n".join(output)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="🎬 Movie Recommender", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Movie Recommendation System\nHybrid SVD + Content-Based · MovieLens 100K")

    with gr.Tabs():
        with gr.TabItem("Known User"):
            gr.Markdown("Enter a user ID (1–943) to get personalized recommendations.")
            with gr.Row():
                user_id = gr.Slider(1, 943, value=1, step=1, label="User ID")
                n_recs = gr.Slider(1, 20, value=10, step=1, label="Number of Recommendations")
                alpha = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Collaborative Weight (α)")
            btn1 = gr.Button("Get Recommendations", variant="primary")
            out1 = gr.Markdown()
            btn1.click(recommend_for_user, inputs=[user_id, n_recs, alpha], outputs=out1)

        with gr.TabItem("Cold Start (New User)"):
            gr.Markdown("Pick movies you've enjoyed — get recommendations without any rating history.")
            seed = gr.Dropdown(ALL_MOVIE_TITLES, multiselect=True, label="Movies You Liked", max_choices=10)
            n_recs2 = gr.Slider(1, 20, value=10, step=1, label="Number of Recommendations")
            btn2 = gr.Button("Get Recommendations", variant="primary")
            out2 = gr.Markdown()
            btn2.click(recommend_coldstart, inputs=[seed, n_recs2], outputs=out2)

    gr.Markdown(
        "---\n"
        "**Dataset:** [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) · "
        "**Source:** [GitHub](https://github.com/YOUR_USERNAME/movie-recommender)"
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
