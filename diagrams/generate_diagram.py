"""
Generate architecture diagram using matplotlib (no external tools needed).
Run: python diagrams/generate_diagram.py
Output: diagrams/architecture.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.png")

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# Color palette
C = {
    "data": "#1e88e5",
    "model": "#43a047",
    "api": "#e53935",
    "monitor": "#fb8c00",
    "deploy": "#8e24aa",
    "arrow": "#90caf9",
    "text": "#ffffff",
    "bg": "#1a1d27",
}


def box(ax, x, y, w, h, label, sublabel="", color="#1e88e5", fontsize=11):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.92,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2 + (0.15 if sublabel else 0), label,
            ha="center", va="center", color="white", fontsize=fontsize, fontweight="bold")
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.22, sublabel,
                ha="center", va="center", color="#cccccc", fontsize=8.5)


def arrow(ax, x1, y1, x2, y2, color="#90caf9"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2))


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(8, 9.5, "Movie Recommendation System — Architecture",
        ha="center", va="center", color="white", fontsize=15, fontweight="bold")

# ── Data layer ────────────────────────────────────────────────────────────────
ax.text(2.5, 8.8, "DATA LAYER", ha="center", color="#90caf9", fontsize=9, style="italic")
box(ax, 0.3, 7.6, 2.1, 0.9, "MovieLens 100K", "100K ratings, 1682 movies", C["data"], fontsize=9)
box(ax, 2.6, 7.6, 2.1, 0.9, "Data Cleaning", "dedup, filter sparse\nusers/items", C["data"], fontsize=9)
box(ax, 4.9, 7.6, 2.1, 0.9, "Feature Eng.", "genre vectors\nyear, popularity", C["data"], fontsize=9)
arrow(ax, 2.4, 8.05, 2.6, 8.05)
arrow(ax, 4.7, 8.05, 4.9, 8.05)

# ── Model layer ───────────────────────────────────────────────────────────────
ax.text(8, 8.8, "MODEL LAYER", ha="center", color="#a5d6a7", fontsize=9, style="italic")
box(ax, 7.2, 7.6, 2.2, 0.9, "SVD (Collab.)", "50 latent factors\nuser/item biases", C["model"], fontsize=9)
box(ax, 9.6, 7.6, 2.2, 0.9, "Content-Based", "TF-IDF + cosine sim\ngenre vectors", C["model"], fontsize=9)
box(ax, 8.4, 6.2, 2.2, 0.9, "Hybrid Blender", "α·CF + (1-α)·CB\ncold-start fallback", C["model"])

arrow(ax, 7.0, 8.0, 7.2, 8.0)  # data → svd
arrow(ax, 7.0, 8.0, 9.6, 8.0)  # data → content
arrow(ax, 8.3, 7.6, 8.9, 7.1)  # svd → hybrid
arrow(ax, 10.2, 7.6, 9.5, 7.1)  # content → hybrid

# ── Evaluation ────────────────────────────────────────────────────────────────
box(ax, 11.8, 7.6, 2.2, 0.9, "Evaluation", "RMSE, MAE\nPrec@K, NDCG@K", "#5c6bc0", fontsize=9)
arrow(ax, 9.5, 8.05, 11.8, 8.05)

# ── API layer ─────────────────────────────────────────────────────────────────
ax.text(5.5, 5.7, "API LAYER", ha="center", color="#ef9a9a", fontsize=9, style="italic")
box(ax, 0.5, 4.5, 2.2, 0.9, "POST /recommend", "known user\nhybrid recs", C["api"], fontsize=9)
box(ax, 3.0, 4.5, 2.5, 0.9, "POST /recommend\n/coldstart", "seed movies\ncontent-based", C["api"], fontsize=9)
box(ax, 5.8, 4.5, 2.4, 0.9, "GET /movies\n/similar/{id}", "item-item sim", C["api"], fontsize=9)
box(ax, 8.4, 4.5, 1.8, 0.9, "GET /health", "liveness check", C["api"], fontsize=9)
box(ax, 10.4, 4.5, 1.8, 0.9, "GET /metrics", "latency, counts\nmodel stats", C["api"], fontsize=9)

# hybrid → API
arrow(ax, 9.5, 6.2, 9.5, 5.4)
arrow(ax, 9.5, 5.4, 1.6, 5.4)
arrow(ax, 9.5, 5.4, 4.25, 5.4)
arrow(ax, 9.5, 5.4, 7.0, 5.4)

# ── FastAPI wrapper ────────────────────────────────────────────────────────────
box(ax, 4.5, 3.2, 4.0, 0.8, "FastAPI + Uvicorn", "Pydantic validation · CORS · middleware", "#c0392b")
arrow(ax, 6.5, 4.5, 6.5, 4.0)

# ── Monitoring ────────────────────────────────────────────────────────────────
ax.text(13.5, 5.7, "MONITORING", ha="center", color="#ffcc80", fontsize=9, style="italic")
box(ax, 12.5, 4.5, 2.5, 0.9, "Monitor Script", "polls /metrics\nlatency alerts", C["monitor"], fontsize=9)
arrow(ax, 12.2, 4.95, 10.4, 4.95, color="#fb8c00")

# ── Deployment ────────────────────────────────────────────────────────────────
ax.text(8, 2.6, "DEPLOYMENT", ha="center", color="#ce93d8", fontsize=9, style="italic")
box(ax, 2.0, 1.5, 2.5, 0.9, "Docker", "Dockerfile\ndocker-compose", C["deploy"], fontsize=9)
box(ax, 4.8, 1.5, 2.5, 0.9, "Render / Railway\n(free tier)", "git push → live\nauto-deploy", C["deploy"], fontsize=9)
box(ax, 7.6, 1.5, 2.5, 0.9, "Hugging Face\nSpaces", "Gradio demo\nfree GPU/CPU", C["deploy"], fontsize=9)
box(ax, 10.4, 1.5, 2.5, 0.9, "GitHub Actions\nCI/CD", "pytest on PR\nbuild & push", C["deploy"], fontsize=9)

arrow(ax, 6.5, 3.2, 6.5, 2.4)
arrow(ax, 6.5, 2.4, 3.25, 2.4)
arrow(ax, 6.5, 2.4, 6.05, 2.4)
arrow(ax, 6.5, 2.4, 8.85, 2.4)
arrow(ax, 6.5, 2.4, 11.65, 2.4)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C["data"], label="Data Pipeline"),
    mpatches.Patch(color=C["model"], label="ML Models"),
    mpatches.Patch(color=C["api"], label="API Endpoints"),
    mpatches.Patch(color=C["monitor"], label="Monitoring"),
    mpatches.Patch(color=C["deploy"], label="Deployment"),
]
ax.legend(handles=legend_items, loc="lower left", facecolor="#1a1d27",
          labelcolor="white", fontsize=9, framealpha=0.8)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT}")
