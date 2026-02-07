"""Tests for the backtest engine and portfolio manager."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from moneytalks.backtest.engine import BacktestEngine, BacktestResult
from moneytalks.backtest.portfolio import PortfolioManager
from moneytalks.strategy.base import Strategy
from moneytalks.strategy.signals import Signal, StrategyContext
from moneytalks.utils.types import Direction, OrderType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample OHLCV data with a predictable trend."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1D", tz="UTC")
    # Create an uptrend
    np.random.seed(42)
    close = 100 + np.arange(100) * 0.5 + np.random.randn(100) * 2
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, size=100),
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


class AlwaysBuyStrategy(Strategy):
    """Test strategy that buys on bar 5 and sells on bar 50."""

    def __init__(self):
        super().__init__(params={"symbol": "TEST"})
        self.symbol = "TEST"

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        idx = context.current_bar_index
        position = context.get_position(self.symbol)

        if idx == 5 and position is None:
            return Signal(
                direction=Direction.LONG,
                symbol=self.symbol,
                order_type=OrderType.MARKET,
                reason="Buy signal at bar 5",
            )
        if idx == 50 and position is not None:
            return Signal(
                direction=Direction.CLOSE,
                symbol=self.symbol,
                order_type=OrderType.MARKET,
                reason="Sell signal at bar 50",
            )
        return None


class NeverTradeStrategy(Strategy):
    """Test strategy that never trades."""

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        return None


# ---------------------------------------------------------------------------
# PortfolioManager tests
# ---------------------------------------------------------------------------


class TestPortfolioManager:
    def test_initial_state(self):
        pm = PortfolioManager(initial_capital=100_000)
        assert pm.cash == 100_000
        assert len(pm.positions) == 0
        assert len(pm.trades) == 0

    def test_execute_long(self, sample_data: pd.DataFrame):
        pm = PortfolioManager(initial_capital=100_000, slippage_rate=0, commission_rate=0)
        bar = sample_data.iloc[10]

        signal = Signal(
            direction=Direction.LONG,
            symbol="TEST",
            quantity=100,
            order_type=OrderType.MARKET,
        )
        order = pm.execute(signal, bar)

        assert order is not None
        assert order.fill_quantity == 100
        assert pm.cash < 100_000  # Cash spent
        assert "TEST" in pm.positions
        assert pm.positions["TEST"].quantity == 100

    def test_execute_close(self, sample_data: pd.DataFrame):
        pm = PortfolioManager(initial_capital=100_000, slippage_rate=0, commission_rate=0)

        # Open position
        bar_open = sample_data.iloc[10]
        signal_open = Signal(direction=Direction.LONG, symbol="TEST", quantity=100)
        pm.execute(signal_open, bar_open)

        # Close position
        bar_close = sample_data.iloc[50]
        signal_close = Signal(direction=Direction.CLOSE, symbol="TEST")
        pm.execute(signal_close, bar_close)

        assert len(pm.trades) == 1
        assert pm.positions["TEST"].quantity == 0

    def test_commission_deducted(self, sample_data: pd.DataFrame):
        pm = PortfolioManager(
            initial_capital=100_000, commission_rate=0.01, slippage_rate=0
        )
        bar = sample_data.iloc[10]
        signal = Signal(direction=Direction.LONG, symbol="TEST", quantity=10)
        order = pm.execute(signal, bar)

        assert order.commission > 0
        expected_cost = 10 * bar["close"] + order.commission
        assert abs(pm.cash - (100_000 - expected_cost)) < 0.01

    def test_equity_tracking(self, sample_data: pd.DataFrame):
        pm = PortfolioManager(initial_capital=100_000, slippage_rate=0, commission_rate=0)

        for i in range(10):
            bar = sample_data.iloc[i]
            pm.record_equity(bar.name, {"TEST": bar["close"]})

        equity = pm.get_equity_series()
        assert len(equity) == 10
        # All equity should be initial capital (no positions)
        assert all(abs(v - 100_000) < 0.01 for v in equity.values)


# ---------------------------------------------------------------------------
# BacktestEngine tests
# ---------------------------------------------------------------------------


class TestBacktestEngine:
    def test_run_with_trades(self, sample_data: pd.DataFrame):
        engine = BacktestEngine(initial_capital=100_000)
        strategy = AlwaysBuyStrategy()
        result = engine.run(strategy, sample_data, symbol="TEST")

        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "AlwaysBuyStrategy"
        assert len(result.trades) >= 1
        assert not result.equity_series.empty

    def test_run_no_trades(self, sample_data: pd.DataFrame):
        engine = BacktestEngine(initial_capital=100_000)
        strategy = NeverTradeStrategy()
        result = engine.run(strategy, sample_data, symbol="TEST")

        assert isinstance(result, BacktestResult)
        assert len(result.trades) == 0
        # Final value should be close to initial (no trades)
        assert abs(result.final_value - 100_000) < 1.0

    def test_run_empty_data(self):
        engine = BacktestEngine()
        strategy = NeverTradeStrategy()
        with pytest.raises(ValueError, match="empty data"):
            engine.run(strategy, pd.DataFrame(), symbol="TEST")

    def test_equity_curve_length(self, sample_data: pd.DataFrame):
        engine = BacktestEngine(initial_capital=100_000)
        strategy = NeverTradeStrategy()
        result = engine.run(strategy, sample_data, symbol="TEST")

        assert len(result.equity_series) == len(sample_data)
