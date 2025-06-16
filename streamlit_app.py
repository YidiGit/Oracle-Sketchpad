import streamlit as st

st.set_page_config(
    page_title="oracle-bone & zodiac",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="auto",
)

from app.views import router, NAV_KEY, DEFAULT_PAGE

selected = st.sidebar.radio(
    "Navigation",
    list(router),
    index=list(router).index(st.session_state.get(NAV_KEY, DEFAULT_PAGE))
)

st.session_state[NAV_KEY] = selected   # keep state in sync
router[selected]()                     # call the chosen page
