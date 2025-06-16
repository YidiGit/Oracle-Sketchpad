# tests/test_app.py
"""
Lean smoke-tests for the Oracle-Bone & Zodiac demo.

• No fragile API calls – works on streamlit-testing 0.3 + 0.4  
• Does not touch the Google-Sheets code (keeps CI offline-safe)  
• Verifies: page renders, 12 glyph cards, embeddings integrity,
  predict() contract.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

import numpy as np
import pytest
import streamlit.testing.v1 as st_test


import sys
from pathlib import Path


# ──────────────────────────── setup ─────────────────────────────
# Ensure the app directory is in sys.path for imports   
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Import the app's views and model functions
from app import views
from app.models import load_embeddings, predict


# ─────────────────────────────────────────────────────────────
#  Disable file-watcher (avoids torch.classes watchdog bug)
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
# ─────────────────────────────────────────────────────────────


# ---------- helper utilities --------------------------------
def _exceptions(sess: st_test.AppTestSession) -> list[Any]:
    """Return uncaught exceptions list (0.3: exception_list, 0.4: exceptions)."""
    return getattr(sess, "exceptions", getattr(sess, "exception_list", []))


def _markdown_html(sess: st_test.AppTestSession) -> Iterable[str]:
    """Yield raw HTML bodies of every markdown block (API-version agnostic)."""
    for field in ("markdowns", "markdown"):
        if hasattr(sess, field):
            for md in getattr(sess, field):
                yield getattr(md, "body", "")
            return
    # very old/unknown version – empty iterator
    return []


def _render(app_obj: st_test.AppTest, page: str) -> st_test.AppTestSession:
    """Switch nav flag, run a single render, return finished session."""
    app_obj.session_state[views.NAV_KEY] = page
    return app_obj.run()


def _black_img(size: int = 64) -> np.ndarray:
    """Return a dummy all-black RGB image for predict() smoke-test."""
    return np.zeros((size, size, 3), dtype=np.uint8)


# ---------- global fixture -----------------------------------
@pytest.fixture(scope="session")
def app() -> st_test.AppTest:
    """Load Streamlit entry-point once per test session (fast)."""
    return st_test.AppTest.from_file("streamlit_app.py")


# ─────────────────────────────────────────────────────────────
#                       UI smoke-tests
# ─────────────────────────────────────────────────────────────
def test_home_renders_and_shows_12_cards(app):
    sess = _render(app, "Home")
    assert not _exceptions(sess), "Home page raised an exception"

    # each zodiac card embeds an <img width='90'>
    glyph_imgs = [m for m in _markdown_html(sess) if "width='90'" in m]
    assert len(glyph_imgs) == 12, "Expected 12 zodiac glyph images"


def test_drawing_page_renders(app):
    sess = _render(app, "Drawing")
    assert not _exceptions(sess), "Drawing page raised an exception"


def test_feedback_page_renders(app):
    sess = _render(app, "Feedback")
    assert not _exceptions(sess), "Feedback page raised an exception"


# ─────────────────────────────────────────────────────────────
#                 Model / data-integrity smoke-tests
# ─────────────────────────────────────────────────────────────
def test_embeddings_shape_match():
    emb, lbl, _ = load_embeddings()
    assert emb.shape[0] == lbl.shape[0] >= 12, "Mismatch between embeddings and labels"


def test_predict_returns_valid_probs():
    best, scores = predict(_black_img())
    assert (
        len(scores) == 12 and 0.0 <= scores[best] <= 1.0
    ), "predict() returned invalid probability vector"
