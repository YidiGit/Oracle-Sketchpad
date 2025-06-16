from __future__ import annotations
import logging
import numpy as np
import open_clip
import torch
from PIL import Image
import streamlit as st

from .config import get_settings

settings = get_settings()
_LOG = logging.getLogger(__name__)

CODE_NAME = {
    "01": "Dog", "02": "Pig", "03": "Rat", "04": "Ox",
    "05": "Tiger", "06": "Rabbit", "07": "Dragon", "08": "Snake",
    "09": "Horse", "10": "Goat", "11": "Monkey", "12": "Rooster"
}


@st.cache_resource(show_spinner="🔄 Loading ML model…")
def load_model():
    torch.set_num_threads(settings.torch_threads)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    _LOG.info("OpenCLIP loaded")
    return model.eval(), preprocess


@st.cache_resource(show_spinner="🔄 Loading embeddings…")
def load_embeddings():
    emb = np.load(settings.embedding_path)
    lbl = np.load(settings.label_path)
    class_names = sorted(
        [
            d
            for d in settings.dataset_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
    )
    _LOG.info("Embeddings loaded: %d vectors", emb.shape[0])
    return emb, lbl, class_names


def predict(image_data) -> tuple[str, dict[str, float]]:
    """Return best-matching class code and full score dict."""
    model, preprocess = load_model()
    emb, lbl, class_names = load_embeddings()

    img = Image.fromarray(image_data.astype("uint8")).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        query = model.encode_image(tensor).squeeze().numpy()

    sim = (emb @ query) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(query))
    scores = {cls: float(sim[lbl == i].mean()) for i, cls in enumerate(class_names)}
    best = max(scores, key=scores.get)
    return best, scores
