"""SMA Crossover strategy — a classic trend-following example.

Buy when the fast SMA crosses above the slow SMA.
Sell (close position) when the fast SMA crosses below the slow SMA.
"""

from __future__ import annotations

import pandas as pd

from moneytalks.strategy.base import Strategy
from moneytalks.strategy.signals import Signal, StrategyContext
from moneytalks.utils.types import Direction, OrderType


class SMACrossStrategy(Strategy):
    """Dual Simple Moving Average crossover strategy.

    Parameters (via ``params`` dict):
        fast_period (int): Fast SMA window, default 10.
        slow_period (int): Slow SMA window, default 30.
        symbol (str): Ticker symbol to trade, default "AAPL".
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.fast_period: int = self.params.get("fast_period", 10)
        self.slow_period: int = self.params.get("slow_period", 30)
        self.symbol: str = self.params.get("symbol", "AAPL")

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        history = context.history
        if len(history) < self.slow_period:
            return None

        close = history["close"]
        fast_sma = close.rolling(self.fast_period).mean().iloc[-1]
        slow_sma = close.rolling(self.slow_period).mean().iloc[-1]

        # Need at least 2 bars to detect a crossover
        if len(history) < self.slow_period + 1:
            return None

        prev_fast = close.rolling(self.fast_period).mean().iloc[-2]
        prev_slow = close.rolling(self.slow_period).mean().iloc[-2]

        position = context.get_position(self.symbol)

        # Golden cross: fast crosses above slow → go long
        if prev_fast <= prev_slow and fast_sma > slow_sma:
            if position is None or not position.is_long:
                return Signal(
                    direction=Direction.LONG,
                    symbol=self.symbol,
                    order_type=OrderType.MARKET,
                    reason=f"Golden cross: SMA{self.fast_period}={fast_sma:.2f} > SMA{self.slow_period}={slow_sma:.2f}",
                    timestamp=bar.name if hasattr(bar, "name") else None,
                )

        # Death cross: fast crosses below slow → close position
        if prev_fast >= prev_slow and fast_sma < slow_sma:
            if position is not None and position.is_long:
                return Signal(
                    direction=Direction.CLOSE,
                    symbol=self.symbol,
                    order_type=OrderType.MARKET,
                    reason=f"Death cross: SMA{self.fast_period}={fast_sma:.2f} < SMA{self.slow_period}={slow_sma:.2f}",
                    timestamp=bar.name if hasattr(bar, "name") else None,
                )

        return None
