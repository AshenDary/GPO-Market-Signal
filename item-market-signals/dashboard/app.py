"""Streamlit dashboard router with shared frame and top navigation."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
FAVICON_PATH = _ROOT / "dashboard" / "assets" / "strawhat_favicon.png"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from dashboard.components.styling import apply_dashboard_style
from dashboard.components.layout import render_footer


NAV_PAGES = [
    st.Page("pages/overview.py", title="Overview", icon=":material/dashboard:"),
    st.Page("pages/lookup.py", title="Item lookup", icon=":material/search:"),
    st.Page("pages/simulator.py", title="Trade Simulator", icon=":material/swap_horiz:"),
    st.Page("pages/model_insights.py", title="Model Insights", icon=":material/analytics:"),
    st.Page("pages/trend.py", title="Trend", icon=":material/show_chart:"),
    st.Page("pages/historical_signal_lab.py", title="Historical Signal Lab", icon=":material/science:"),
    st.Page("pages/value_list.py", title="Value List", icon=":material/table:"),
    st.Page("pages/guide.py", title="Start Here", icon=":material/info:", default=True),
]

REPO_URL = "https://github.com/AshenDary/GPO_Market.git"


st.set_page_config(page_title="GPO Item Market Signals", page_icon=str(FAVICON_PATH), layout="wide")
apply_dashboard_style()

st.markdown(
    dedent(
        """
    <div class="app-shell">
        <div class="app-kicker">Grand Piece Online market intelligence</div>
        <h1 class="app-title">Item Market Signals</h1>
        <p class="app-subtitle">
            Fair values, confidence bands, tier context, and snapshot trends
            from the latest tracked market data.
        </p>
    </div>
    """,
    ).strip(),
    unsafe_allow_html=True,
)

page = st.navigation(NAV_PAGES, position="top")
page.run()
render_footer(REPO_URL)
