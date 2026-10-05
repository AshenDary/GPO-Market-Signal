from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from market_signals.evaluator import evaluate


RUNNER = CliRunner()


def _feature_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "Low Signal Sword",
                "shortcut": "LSS",
                "join_key": "low signal sword",
                "value": 100.0,
                "ci_low": 80.0,
                "ci_high": 120.0,
                "confidence": "low",
                "trade_count": 52,
                "demand": "Low Demand",
            }
        ]
    )


def test_cli_describes_trade_count_as_activity_signal(monkeypatch) -> None:
    monkeypatch.setattr(evaluate, "_load_feature_matrix", lambda: _feature_matrix())
    monkeypatch.setattr(evaluate, "compute_trend", lambda _join_key: None)

    result = RUNNER.invoke(evaluate.app, ["Low Signal Sword"])

    assert result.exit_code == 0
    assert "Activity signal : trade_count=52" in result.output
    assert "thin trade_count activity signal" in result.output
    assert "trades observed" not in result.output
