"""Historical signal lab dashboard page."""

from __future__ import annotations

import streamlit as st

from dashboard.components.data import load_historical_signal_lab
from dashboard.components.layout import render_refresh_button
from dashboard.components.views import render_historical_signal_lab


render_refresh_button("refresh_historical_signal_lab")

try:
    historical_signal_result = load_historical_signal_lab()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

render_historical_signal_lab(historical_signal_result)
