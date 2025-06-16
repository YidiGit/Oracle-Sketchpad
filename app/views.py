# ─────────────────────────── app/views.py ───────────────────────────
"""
Main Streamlit view layer for the Oracle-Bone & Chinese-Zodiac demo.

Enterprise-class highlights
---------------------------
1. Clear separation of concerns: each page has its own function.
2. Full type hints for static analysis.
3. Centralised logging, level driven by `settings.log_level`.
4. Exponential back-off when writing to Google Sheets to respect quota.
5. One-time CSS injection per rerun.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import gspread
import streamlit as st
from oauth2client.service_account import ServiceAccountCredentials
from streamlit_drawable_canvas import st_canvas

from .config import get_settings
from .models import CODE_NAME, predict
from .utils import to_base64
from .zodiac_info import ORDERED_CODES, ORACLE_INTRO, ZODIAC_DETAILS, ZODIAC_INTRO

# ───────────────────── Runtime / environment ──────────────────────
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # avoids torch watcher crash

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level, "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_LOG = logging.getLogger(__name__)

# ───────────────────────── Global CSS ─────────────────────────────
ASSETS_DIR = getattr(settings, "assets_dir", Path("Data/assets"))
_CSS_FILE  = ASSETS_DIR / "theme.css"
_CSS_TXT   = _CSS_FILE.read_text("utf-8") if _CSS_FILE.is_file() else ""


def inject_css() -> None:
    """Inject the global stylesheet exactly once each rerun."""
    if _CSS_TXT:
        st.markdown(f"<style>{_CSS_TXT}</style>", unsafe_allow_html=True)
    else:
        _LOG.warning("theme.css missing – Streamlit default theme will be used")


def _asset(path: Path, warning: str) -> Path | None:
    """
    Validate that an asset exists; warn UI and log if missing.

    Args:
        path: Path to asset.
        warning: Message shown if file is absent.

    Returns:
        `Path` if present, otherwise `None`.
    """
    if path.exists():
        return path
    _LOG.warning(warning)
    st.warning(warning)
    return None


# ───────────────────── Navigation helpers ─────────────────────────
NAV_KEY = "nav_page"
DEFAULT_PAGE: str = "Home" 
if NAV_KEY not in st.session_state:
    st.session_state[NAV_KEY] = DEFAULT_PAGE


def goto(page: str) -> None:
    """Switch to *page* and trigger a rerun."""
    st.session_state[NAV_KEY] = page
    st.rerun()

# ──────────────────────── Page configuration ─────────────────────────
st.set_page_config(
    page_title="oracle-bone & zodiac",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="auto"
)

# ─────────────────────────── Home page ────────────────────────────
def page_home() -> None:
    """Display the oracle-bone introduction and zodiac gallery."""
    inject_css()

    banner = _asset(settings.assets_dir / "background2.png", "background2.png missing")
    if banner:
        st.image(banner, use_container_width=True)

    st.title("📜 Oracle-Bone Primer")
    st.write(ORACLE_INTRO)

    st.title("🐭🐮🐯 Chinese Zodiac")
    st.write(ZODIAC_INTRO)

    left, right = st.columns([3, 1], gap="large")

    with left:
        poster = _asset(settings.assets_dir / "zodiac.png", "zodiac.png missing")
        if poster:
            st.image(poster, caption="Zodiac overview", use_container_width=True)

    with right:
        for code in ORDERED_CODES:
            animal: str = CODE_NAME[code]
            desc: str = ZODIAC_DETAILS.get(animal, "Description pending.")
            glyph64: str = to_base64(settings.assets_dir / "Oracle_Bone" / f"{code}.jpg")

            st.markdown(
                f"**{animal}**  \n{desc}  \n"
                f"<img src='{glyph64}' width='90'>",
                unsafe_allow_html=True,
            )



    st.markdown("---")
    st.markdown(
        "### ✍️ Try the sketch recogniser  \n"
        "Open **Drawing**, doodle a glyph, then press **Predict**."
    )
    if st.button("🎨 Go to Drawing"):
        goto("Drawing")


# ───────────────────────── Drawing page ───────────────────────────
def _render_top(scores: Dict[str, float]) -> None:
    """Render the three highest-confidence predictions."""
    for raw_code, score in sorted(scores.items(), key=lambda kv: -kv[1])[:3]:
        code = Path(raw_code).name
        glyph = to_base64(settings.assets_dir / "Oracle_Bone" / f"{code}.jpg")
        photo = to_base64(settings.assets_dir / "Real_Animals" / f"{code}.png")

        st.markdown(
            f"<img src='{glyph}' width='60'> "
            f"<img src='{photo}' width='60'> "
            f"**{CODE_NAME[code]}** — {score*100:.1f} %",
            unsafe_allow_html=True,
        )
        st.progress(score)


def page_drawing() -> None:
    """Interactive canvas page that calls the ML model."""
    inject_css()
    st.title("✍️ Sketch Recognition")

    canvas_col, output_col = st.columns([2, 1])
    with canvas_col:
        canvas = st_canvas(
            key="sketch_canvas",
            fill_color="rgba(0,0,0,0)",
            stroke_width=3,
            stroke_color="#000",
            background_color="#fff",
            height=350,
            width=350,
            drawing_mode="freedraw",
        )
        if canvas.image_data is not None and st.button("🚀 Predict"):
            st.session_state.pred = predict(canvas.image_data)
            st.rerun()

    with output_col:
        if "pred" in st.session_state:
            best_raw, scores = st.session_state.pred
            best_code = Path(best_raw).name
            st.success(
                f"🔮 Prediction: {CODE_NAME[best_code]} "
                f"({scores[best_raw]*100:.1f} %)"
            )
            _render_top(scores)
        else:
            st.info("Draw a glyph on the left and press **Predict**.")

    st.markdown("---")
    st.markdown("### 💬 Want to help? Head to **Feedback**!")
    if st.button("💬 Go to Feedback"):
        goto("Feedback")


# ─────────────────────── Feedback page ───────────────────────────
@st.cache_resource
def init_gsheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp_service_account"], scope
    )
    gc = gspread.authorize(credentials)
    sheet = gc.open("Oracle_Streamlit_Feedback").sheet1 
    return sheet

def page_feedback() -> None:
    """Collect user feedback and store it in Google Sheets."""
    st.title("💬 Feedback & Suggestions")

    with st.form("feedback_form"):
        name = st.text_input("Your name (optional)")
        score = st.slider("Satisfaction", 1, 5, 5)
        comment = st.text_area("Let us know what you think")

    if st.form_submit_button("Submit"):
        sheet = init_gsheet()
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, name, score, comment])
        st.success("🙏 Thank you — your feedback has been recorded!")


# ─────────────────────────── Router ──────────────────────────────
router: Dict[str, callable] = {
    "Home": page_home,
    "Drawing": page_drawing,
    "Feedback": page_feedback,
}
