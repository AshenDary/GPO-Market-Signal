"""Historical signal lab dashboard page."""

from __future__ import annotations

import streamlit as st

from dashboard.components.data import load_historical_signal_lab, load_snapshot_quality_preflight
from dashboard.components.layout import render_refresh_button
from dashboard.components.views import render_historical_signal_lab, render_snapshot_quality_preflight


render_refresh_button("refresh_historical_signal_lab")

try:
    snapshot_quality_result = load_snapshot_quality_preflight()
except (FileNotFoundError, ValueError) as exc:
    st.warning(f"Snapshot quality preflight unavailable: {exc}")
else:
    render_snapshot_quality_preflight(snapshot_quality_result)

try:
    historical_signal_result = load_historical_signal_lab()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

render_historical_signal_lab(historical_signal_result)
