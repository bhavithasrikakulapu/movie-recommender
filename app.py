import os, subprocess, sys

MODEL_DIR = "models"
SVD_PATH = os.path.join(MODEL_DIR, "svd_model.pkl")

if not os.path.exists(SVD_PATH):
    print("First boot — downloading data and training models...")
    subprocess.run([sys.executable, "data/download_data.py"], check=True)
    subprocess.run([sys.executable, "data/preprocess.py"], check=True)
    subprocess.run([sys.executable, "models/train.py"], check=True)
    print("Training complete.")

from app_gradio import demo
demo.launch()
