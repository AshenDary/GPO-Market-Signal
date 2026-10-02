"""Start-here landing page for first-time dashboard visitors."""

from __future__ import annotations

import streamlit as st


st.markdown("<h2 class='section-title'>Start Here</h2>", unsafe_allow_html=True)

st.markdown(
    """
This tool estimates fair value for a Grand Piece Online trade item, shows a
confidence band around that estimate, adds trend context when enough history
exists, and gives a plain verdict: good deal, fair, or overpriced.
"""
)

st.markdown("<h2 class='section-title'>Explore the dashboard</h2>", unsafe_allow_html=True)
st.markdown("Choose a view below to jump directly into the part of the market you want to inspect.")

directory = (
    (
        "Overview",
        "See the current catalog at a glance, including coverage, confidence, and the most-traded items.",
        "pages/overview.py",
        ":material/dashboard:",
    ),
    (
        "Item Lookup",
        "Check one item, compare an asking price with its estimated fair value, and review the confidence band.",
        "pages/lookup.py",
        ":material/search:",
    ),
    (
        "Trend",
        "Inspect how tracked values changed across dated snapshots when enough history is available.",
        "pages/trend.py",
        ":material/show_chart:",
    ),
    (
        "Market Change Alerts",
        "Review latest snapshot items whose value, demand ratio, or activity signal moved unusually.",
        "pages/market_change_alerts.py",
        ":material/notification_important:",
    ),
    (
        "Historical Signal Lab",
        "Compare simple past snapshot signals with later gpovalues published-estimate changes.",
        "pages/historical_signal_lab.py",
        ":material/science:",
    ),
    (
        "Value List",
        "Search the full available value catalog when you want to compare many items or discover a price range.",
        "pages/value_list.py",
        ":material/table:",
    ),
    (
        "Trade Simulator",
        "Compare the estimated value of items on both sides of a trade and surface confidence caveats.",
        "pages/simulator.py",
        ":material/swap_horiz:",
    ),
    (
        "Model Insights",
        "Review the structural model's diagnostics to understand how tier and category features inform estimates.",
        "pages/model_insights.py",
        ":material/analytics:",
    ),
)

directory_columns = st.columns(2)
for index, (title, description, page, icon) in enumerate(directory):
    with directory_columns[index % 2]:
        with st.container(border=True):
            st.markdown(
                f"<h3 class='item-title'>{title}</h3><p class='item-meta'>{description}</p>",
                unsafe_allow_html=True,
            )
            st.page_link(page, label=f"Open {title}", icon=icon, width="stretch")

st.markdown("<h2 class='section-title'>How the signals work</h2>", unsafe_allow_html=True)

st.markdown(
    """

### Data sources

- Primary source: [gpovalues.com](https://gpovalues.com), which publishes solved
  item values from observed Discord trades.
- Method details: [gpovalues methodology](https://gpovalues.com/legal/methodology).
- This project reads their public API and does not scrape Discord directly.
- Secondary source: a curated tier-list JSON used for structural context
  (category, rarity, obtainability). It does not provide market prices.

### Confidence labels

The confidence label comes from gpovalues trade-count bands and is shown as
low, medium, or high. Low-confidence values are still useful, but should be
read as directional rather than exact.

### Trend interpretation

Trend compares item value across dated snapshots. It only appears when there
are multiple snapshot dates, so some items may still show "not enough snapshot
history yet" while data continues to accumulate.

### Market Change Alerts

Market Change Alerts compare the latest gpovalues snapshot with the immediately
previous available snapshot. They are leads for review, not buy or sell
recommendations. The page groups fired rows by value movement, demand-ratio
change, and `trade_count` activity change, then keeps non-alert guard rows
visible for insufficient history, missing comparison items, duplicate item
identities, stale snapshots, and invalid or extreme values. `trade_count` is an
activity indicator that can decrease; it is not daily trade volume or new
trades today.

### Historical Signal Lab

Historical Signal Lab evaluates simple item-date signals from past gpovalues
snapshots: prior 7-day published-value change, prior 7-day demand-ratio
change when both ratios are valid, and confidence category on the evaluation
date. Outcomes are the same item's later gpovalues published estimate about
14 days later, not resale profit, asking-price verdicts, or completed trades.

### Coverage limits

Only items that gpovalues has price data for appear in this dashboard. This is
not a full catalog of every item in the game.

### Where to go next

Use Overview for the current catalog, Item lookup to check a specific
item/asking price, and Trend to inspect value history over time.
"""
)
