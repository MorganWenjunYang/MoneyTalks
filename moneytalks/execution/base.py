"""Abstract base class for broker / order execution."""

from __future__ import annotations

from abc import ABC, abstractmethod

from moneytalks.strategy.signals import Order, Position, Signal


class Broker(ABC):
    """Abstract broker interface for order execution.

    Implementations can target paper trading (simulated) or live
    broker APIs (Alpaca, Interactive Brokers, etc.).
    """

    @abstractmethod
    def submit_order(self, signal: Signal) -> Order:
        """Submit an order based on a trade signal.

        Args:
            signal: The trade signal to execute.

        Returns:
            The resulting Order with fill information.
        """
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Position | None:
        """Get the current position for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Position object, or None if no position exists.
        """
        ...

    @abstractmethod
    def get_balance(self) -> float:
        """Get the current cash balance.

        Returns:
            Available cash balance.
        """
        ...

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Get the total portfolio value (cash + positions).

        Returns:
            Total portfolio value.
        """
        ...
