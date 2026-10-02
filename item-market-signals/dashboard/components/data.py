"""Shared Streamlit data loaders that reuse the package pipeline directly."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from market_signals.analysis.historical_signal_lab import (
    HistoricalSignalResult,
    run_historical_signal_lab,
)
from market_signals.analysis.market_change_alerts import (
    MarketChangeAlertResult,
    run_market_change_alerts,
)
from market_signals.features.build_feature_matrix import build_feature_matrix
from market_signals.models.trend_model import load_snapshot_history, most_traded
from market_signals.models.value_regression import (
    ValueRegressionResult,
    build_value_explainer,
    load_latest_tier_reference,
    train_value_regression,
)

CACHE_TTL_SECONDS = 60 * 60


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_feature_matrix() -> pd.DataFrame:
    """Build the current feature matrix from the latest available snapshots."""
    return build_feature_matrix()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_history() -> pd.DataFrame:
    """Load all gpovalues snapshots for trend views."""
    return load_snapshot_history()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_historical_signal_lab() -> HistoricalSignalResult:
    """Run the historical signal lab from raw gpovalues snapshots."""
    return run_historical_signal_lab()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_market_change_alerts() -> MarketChangeAlertResult:
    """Run current-vs-previous market change alerts from raw snapshots."""
    return run_market_change_alerts()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_most_traded(limit: int = 15) -> pd.DataFrame:
    """Load the current most actively traded item ranking."""
    return most_traded(limit=limit)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_tier_reference() -> pd.DataFrame:
    """Load the latest structural tier-reference snapshot."""
    return load_latest_tier_reference()


@st.cache_resource(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_value_regression_model() -> ValueRegressionResult:
    """Train and cache the structural value model object."""
    return train_value_regression()


@st.cache_resource(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_value_explainer(_value_model: ValueRegressionResult) -> object:
    """Build and cache the SHAP explainer adjacent to the trained model."""
    return build_value_explainer(_value_model)
