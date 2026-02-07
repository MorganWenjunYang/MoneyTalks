"""Signal and context data classes for the strategy layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from moneytalks.utils.types import Direction, OrderType


@dataclass
class Signal:
    """A trade signal emitted by a strategy.

    Attributes:
        direction: LONG, SHORT, or CLOSE.
        symbol: Ticker symbol for the trade.
        quantity: Number of shares/contracts. 0 means the portfolio manager decides.
        price: Target price (None = market price).
        order_type: MARKET or LIMIT.
        reason: Human-readable explanation of why this signal was generated.
        timestamp: When the signal was generated.
    """

    direction: Direction
    symbol: str
    quantity: float = 0.0
    price: float | None = None
    order_type: OrderType = OrderType.MARKET
    reason: str = ""
    timestamp: datetime | None = None


@dataclass
class Position:
    """Represents a current position in a single instrument.

    Attributes:
        symbol: Ticker symbol.
        quantity: Signed quantity (positive = long, negative = short).
        avg_entry_price: Volume-weighted average entry price.
        entry_time: When the position was first opened.
    """

    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    entry_time: datetime | None = None

    @property
    def is_open(self) -> bool:
        return self.quantity != 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def market_value(self) -> float:
        """Unsigned market value at entry price."""
        return abs(self.quantity) * self.avg_entry_price


@dataclass
class Order:
    """Represents a filled or pending order.

    Attributes:
        signal: The originating signal.
        fill_price: Actual execution price (including slippage).
        fill_quantity: Actual filled quantity.
        commission: Commission charged.
        fill_time: When the order was filled.
    """

    signal: Signal
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    commission: float = 0.0
    fill_time: datetime | None = None


@dataclass
class StrategyContext:
    """Context object passed to strategies on each bar.

    Provides read-only access to historical data, current positions,
    and account state. This prevents strategies from directly
    manipulating engine internals.
    """

    data: pd.DataFrame = field(repr=False)
    initial_capital: float = 100_000.0

    # Mutable state updated by the engine each bar
    current_bar_index: int = 0
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)

    def __post_init__(self):
        if self.cash == 0.0:
            self.cash = self.initial_capital

    @property
    def current_bar(self) -> pd.Series:
        """Return the current bar as a Series."""
        return self.data.iloc[self.current_bar_index]

    @property
    def history(self) -> pd.DataFrame:
        """Return all bars up to and including the current bar."""
        return self.data.iloc[: self.current_bar_index + 1]

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value = cash + sum of open position market values."""
        position_value = sum(
            pos.quantity * self.current_bar.get("close", pos.avg_entry_price)
            for pos in self.positions.values()
            if pos.is_open
        )
        return self.cash + position_value

    def get_position(self, symbol: str) -> Position | None:
        """Get the current position for a symbol, or None."""
        pos = self.positions.get(symbol)
        if pos and pos.is_open:
            return pos
        return None

    def update(self, bar_index: int, cash: float, positions: dict[str, Position]):
        """Update context state (called by the engine each bar)."""
        self.current_bar_index = bar_index
        self.cash = cash
        self.positions = positions
