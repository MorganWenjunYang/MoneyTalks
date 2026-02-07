"""Tests for the metrics calculator."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from moneytalks.backtest.metrics import MetricsCalculator, MetricsReport
from moneytalks.backtest.portfolio import TradeRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator() -> MetricsCalculator:
    return MetricsCalculator()


@pytest.fixture
def profitable_equity() -> pd.Series:
    """Create an equity curve with a steady uptrend."""
    dates = pd.date_range("2024-01-01", periods=252, freq="1D", tz="UTC")
    # Steady 20% annual return
    values = 100_000 * (1 + 0.20 / 252) ** np.arange(252)
    return pd.Series(values, index=dates, name="equity")


@pytest.fixture
def losing_equity() -> pd.Series:
    """Create an equity curve with a steady downtrend."""
    dates = pd.date_range("2024-01-01", periods=252, freq="1D", tz="UTC")
    values = 100_000 * (1 - 0.15 / 252) ** np.arange(252)
    return pd.Series(values, index=dates, name="equity")


@pytest.fixture
def sample_trades() -> list[TradeRecord]:
    """Create sample trades: 3 winners and 2 losers."""
    base_time = datetime(2024, 1, 1)
    trades = [
        TradeRecord(
            symbol="AAPL", direction="LONG",
            entry_time=base_time,
            entry_price=150.0,
            exit_time=base_time + timedelta(days=5),
            exit_price=160.0,
            quantity=100, pnl=1000.0, pnl_pct=0.0667, commission=15.0,
        ),
        TradeRecord(
            symbol="AAPL", direction="LONG",
            entry_time=base_time + timedelta(days=10),
            entry_price=155.0,
            exit_time=base_time + timedelta(days=15),
            exit_price=145.0,
            quantity=100, pnl=-1000.0, pnl_pct=-0.0645, commission=15.0,
        ),
        TradeRecord(
            symbol="AAPL", direction="LONG",
            entry_time=base_time + timedelta(days=20),
            entry_price=148.0,
            exit_time=base_time + timedelta(days=30),
            exit_price=165.0,
            quantity=100, pnl=1700.0, pnl_pct=0.1149, commission=15.0,
        ),
        TradeRecord(
            symbol="AAPL", direction="LONG",
            entry_time=base_time + timedelta(days=35),
            entry_price=162.0,
            exit_time=base_time + timedelta(days=38),
            exit_price=158.0,
            quantity=100, pnl=-400.0, pnl_pct=-0.0247, commission=15.0,
        ),
        TradeRecord(
            symbol="AAPL", direction="LONG",
            entry_time=base_time + timedelta(days=40),
            entry_price=160.0,
            exit_time=base_time + timedelta(days=50),
            exit_price=175.0,
            quantity=100, pnl=1500.0, pnl_pct=0.0938, commission=15.0,
        ),
    ]
    return trades


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMetricsCalculator:
    def test_annual_return_positive(
        self, calculator: MetricsCalculator, profitable_equity: pd.Series, sample_trades
    ):
        metrics = calculator.calculate(
            profitable_equity, sample_trades, initial_capital=100_000
        )
        assert metrics.annual_return > 0
        assert metrics.annual_return == pytest.approx(0.20, abs=0.05)

    def test_annual_return_negative(
        self, calculator: MetricsCalculator, losing_equity: pd.Series, sample_trades
    ):
        metrics = calculator.calculate(
            losing_equity, sample_trades, initial_capital=100_000
        )
        assert metrics.annual_return < 0

    def test_sharpe_ratio_positive(
        self, calculator: MetricsCalculator, profitable_equity: pd.Series, sample_trades
    ):
        metrics = calculator.calculate(
            profitable_equity, sample_trades, initial_capital=100_000
        )
        assert metrics.sharpe_ratio > 0

    def test_max_drawdown(
        self, calculator: MetricsCalculator, profitable_equity: pd.Series, sample_trades
    ):
        metrics = calculator.calculate(
            profitable_equity, sample_trades, initial_capital=100_000
        )
        assert 0 <= metrics.max_drawdown <= 1.0

    def test_win_rate(
        self, calculator: MetricsCalculator, profitable_equity: pd.Series, sample_trades
    ):
        metrics = calculator.calculate(
            profitable_equity, sample_trades, initial_capital=100_000
        )
        # 3 winners out of 5
        assert metrics.win_rate == pytest.approx(0.6, abs=0.01)
        assert metrics.total_trades == 5

    def test_profit_loss_ratio(
        self, calculator: MetricsCalculator, profitable_equity: pd.Series, sample_trades
    ):
        metrics = calculator.calculate(
            profitable_equity, sample_trades, initial_capital=100_000
        )
        # Avg win = (1000+1700+1500)/3 = 1400, Avg loss = (1000+400)/2 = 700
        assert metrics.profit_loss_ratio == pytest.approx(2.0, abs=0.01)

    def test_avg_holding_period(
        self, calculator: MetricsCalculator, profitable_equity: pd.Series, sample_trades
    ):
        metrics = calculator.calculate(
            profitable_equity, sample_trades, initial_capital=100_000
        )
        # Average of 5, 5, 10, 3, 10 days = 6.6 days
        assert metrics.avg_holding_period == pytest.approx(6.6, abs=0.1)

    def test_total_commission(
        self, calculator: MetricsCalculator, profitable_equity: pd.Series, sample_trades
    ):
        metrics = calculator.calculate(
            profitable_equity, sample_trades, initial_capital=100_000
        )
        assert metrics.total_commission == pytest.approx(75.0, abs=0.01)

    def test_empty_equity(self, calculator: MetricsCalculator):
        metrics = calculator.calculate(
            pd.Series(dtype=float), [], initial_capital=100_000
        )
        assert metrics.annual_return == 0.0
        assert metrics.total_trades == 0

    def test_no_trades(
        self, calculator: MetricsCalculator, profitable_equity: pd.Series
    ):
        metrics = calculator.calculate(
            profitable_equity, [], initial_capital=100_000
        )
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0

    def test_summary_format(
        self, calculator: MetricsCalculator, profitable_equity: pd.Series, sample_trades
    ):
        metrics = calculator.calculate(
            profitable_equity, sample_trades, initial_capital=100_000
        )
        summary = metrics.summary()
        assert "BACKTEST PERFORMANCE REPORT" in summary
        assert "Annual Return" in summary
        assert "Sharpe Ratio" in summary
        assert "Max Drawdown" in summary

    def test_to_dict(
        self, calculator: MetricsCalculator, profitable_equity: pd.Series, sample_trades
    ):
        metrics = calculator.calculate(
            profitable_equity, sample_trades, initial_capital=100_000
        )
        d = metrics.to_dict()
        assert "annual_return" in d
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d
        assert len(d) == 12
