"""Market change alerts dashboard page."""

from __future__ import annotations

import streamlit as st

from dashboard.components.data import load_market_change_alerts
from dashboard.components.layout import render_refresh_button
from dashboard.components.views import render_market_change_alerts


render_refresh_button("refresh_market_change_alerts")

try:
    market_change_result = load_market_change_alerts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

render_market_change_alerts(market_change_result)
