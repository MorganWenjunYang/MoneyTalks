"""Abstract base class for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from moneytalks.strategy.signals import Signal, StrategyContext


class Strategy(ABC):
    """Base class for all trading strategies.

    To create a strategy, subclass this and implement ``on_bar``.
    Optionally override ``on_init`` for pre-computation of indicators.

    Example::

        class MyStrategy(Strategy):
            def on_bar(self, bar, context):
                if bar["close"] > bar["open"]:
                    return Signal(Direction.LONG, symbol="AAPL")
                return None
    """

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    @property
    def name(self) -> str:
        """Strategy name (defaults to class name)."""
        return self.__class__.__name__

    def on_init(self, context: StrategyContext) -> None:
        """Called once before the backtest loop starts.

        Override to pre-compute indicators or set up internal state.

        Args:
            context: Strategy context with full historical data.
        """
        pass

    @abstractmethod
    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        """Called on every bar during backtesting / live execution.

        Args:
            bar: Current bar data (open, high, low, close, volume).
            context: Strategy context with history, positions, cash.

        Returns:
            A Signal if the strategy wants to trade, or None.
        """
        ...
