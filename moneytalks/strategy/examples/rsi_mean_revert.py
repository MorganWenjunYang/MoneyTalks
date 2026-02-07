"""RSI Mean Reversion strategy.

Buy when RSI drops below the oversold threshold (default 30).
Sell (close) when RSI rises above the overbought threshold (default 70).
"""

from __future__ import annotations

import pandas as pd

from moneytalks.strategy.base import Strategy
from moneytalks.strategy.signals import Signal, StrategyContext
from moneytalks.utils.types import Direction, OrderType


class RSIMeanRevertStrategy(Strategy):
    """RSI-based mean reversion strategy.

    Parameters (via ``params`` dict):
        rsi_period (int): RSI lookback window, default 14.
        oversold (float): RSI level to trigger buy, default 30.
        overbought (float): RSI level to trigger sell, default 70.
        symbol (str): Ticker symbol to trade, default "AAPL".
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.rsi_period: int = self.params.get("rsi_period", 14)
        self.oversold: float = self.params.get("oversold", 30.0)
        self.overbought: float = self.params.get("overbought", 70.0)
        self.symbol: str = self.params.get("symbol", "AAPL")

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        history = context.history
        if len(history) < self.rsi_period + 1:
            return None

        rsi = self._compute_rsi(history["close"], self.rsi_period)
        if rsi is None:
            return None

        position = context.get_position(self.symbol)

        # Oversold → buy
        if rsi < self.oversold and (position is None or not position.is_long):
            return Signal(
                direction=Direction.LONG,
                symbol=self.symbol,
                order_type=OrderType.MARKET,
                reason=f"RSI oversold: RSI({self.rsi_period})={rsi:.1f} < {self.oversold}",
                timestamp=bar.name if hasattr(bar, "name") else None,
            )

        # Overbought → sell / close
        if rsi > self.overbought and position is not None and position.is_long:
            return Signal(
                direction=Direction.CLOSE,
                symbol=self.symbol,
                order_type=OrderType.MARKET,
                reason=f"RSI overbought: RSI({self.rsi_period})={rsi:.1f} > {self.overbought}",
                timestamp=bar.name if hasattr(bar, "name") else None,
            )

        return None

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int) -> float | None:
        """Compute the RSI value for the latest bar."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
        avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]

        if pd.isna(avg_gain) or pd.isna(avg_loss):
            return None
        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
