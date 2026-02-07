"""Portfolio management: position tracking, order execution, commission/slippage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from moneytalks.config import DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE_RATE
from moneytalks.strategy.signals import Order, Position, Signal
from moneytalks.utils.logger import get_logger
from moneytalks.utils.types import Direction

logger = get_logger("backtest.portfolio")


@dataclass
class TradeRecord:
    """A completed round-trip trade (entry + exit)."""

    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0


class PortfolioManager:
    """Manages cash, positions, and trade execution during backtesting.

    Supports:
        - Long / short / close operations
        - Configurable commission rate and slippage rate
        - Full trade history tracking
        - Equity curve recording
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # Current open positions: symbol -> Position
        self.positions: dict[str, Position] = {}

        # Completed trade records
        self.trades: list[TradeRecord] = []

        # Equity curve: list of (timestamp, portfolio_value)
        self.equity_curve: list[tuple[datetime, float]] = []

        # All executed orders
        self.orders: list[Order] = []

    def execute(self, signal: Signal, bar: pd.Series) -> Order | None:
        """Execute a trade signal against the current bar.

        Args:
            signal: The trade signal to execute.
            bar: Current bar data (must have 'close' and a datetime index name).

        Returns:
            The filled Order, or None if execution was skipped.
        """
        close_price = bar["close"]
        timestamp = bar.name  # DatetimeIndex value

        if signal.direction == Direction.LONG:
            return self._open_long(signal, close_price, timestamp)
        elif signal.direction == Direction.SHORT:
            return self._open_short(signal, close_price, timestamp)
        elif signal.direction == Direction.CLOSE:
            return self._close_position(signal, close_price, timestamp)
        return None

    def record_equity(self, timestamp: datetime, current_price_map: dict[str, float]):
        """Record the current portfolio value for the equity curve.

        Args:
            timestamp: Current bar timestamp.
            current_price_map: symbol -> current close price.
        """
        value = self.cash
        for symbol, pos in self.positions.items():
            if pos.is_open:
                price = current_price_map.get(symbol, pos.avg_entry_price)
                value += pos.quantity * price
        self.equity_curve.append((timestamp, value))

    def get_equity_series(self) -> pd.Series:
        """Return the equity curve as a pandas Series."""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        timestamps, values = zip(*self.equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="equity")

    # ------------------------------------------------------------------
    # Private execution methods
    # ------------------------------------------------------------------

    def _open_long(
        self, signal: Signal, price: float, timestamp: datetime
    ) -> Order | None:
        """Open or add to a long position."""
        fill_price = price * (1 + self.slippage_rate)  # Buy at slightly higher price

        # Determine quantity
        quantity = signal.quantity
        if quantity <= 0:
            # Use all available cash (simple sizing)
            quantity = int(self.cash * 0.95 / fill_price)
        if quantity <= 0:
            logger.warning(f"Insufficient cash to open long on {signal.symbol}")
            return None

        cost = quantity * fill_price
        commission = cost * self.commission_rate

        if cost + commission > self.cash:
            # Reduce quantity to fit budget
            quantity = int((self.cash * 0.95) / (fill_price * (1 + self.commission_rate)))
            if quantity <= 0:
                logger.warning(f"Insufficient cash to open long on {signal.symbol}")
                return None
            cost = quantity * fill_price
            commission = cost * self.commission_rate

        # Close any existing short first
        existing = self.positions.get(signal.symbol)
        if existing and existing.is_short:
            self._close_position(signal, price, timestamp)

        self.cash -= cost + commission

        # Update position
        pos = self.positions.get(signal.symbol)
        if pos and pos.is_long:
            # Average into existing position
            total_qty = pos.quantity + quantity
            pos.avg_entry_price = (
                pos.avg_entry_price * pos.quantity + fill_price * quantity
            ) / total_qty
            pos.quantity = total_qty
        else:
            self.positions[signal.symbol] = Position(
                symbol=signal.symbol,
                quantity=quantity,
                avg_entry_price=fill_price,
                entry_time=timestamp,
            )

        order = Order(
            signal=signal,
            fill_price=fill_price,
            fill_quantity=quantity,
            commission=commission,
            fill_time=timestamp,
        )
        self.orders.append(order)
        logger.debug(
            f"LONG {signal.symbol} qty={quantity} @ {fill_price:.2f} "
            f"commission={commission:.2f}"
        )
        return order

    def _open_short(
        self, signal: Signal, price: float, timestamp: datetime
    ) -> Order | None:
        """Open or add to a short position."""
        fill_price = price * (1 - self.slippage_rate)  # Sell at slightly lower price

        quantity = signal.quantity
        if quantity <= 0:
            quantity = int(self.cash * 0.95 / fill_price)
        if quantity <= 0:
            logger.warning(f"Insufficient margin to open short on {signal.symbol}")
            return None

        proceeds = quantity * fill_price
        commission = proceeds * self.commission_rate

        # Close any existing long first
        existing = self.positions.get(signal.symbol)
        if existing and existing.is_long:
            self._close_position(signal, price, timestamp)

        # For shorts, we receive proceeds but need margin
        self.cash += proceeds - commission

        self.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            quantity=-quantity,
            avg_entry_price=fill_price,
            entry_time=timestamp,
        )

        order = Order(
            signal=signal,
            fill_price=fill_price,
            fill_quantity=quantity,
            commission=commission,
            fill_time=timestamp,
        )
        self.orders.append(order)
        logger.debug(
            f"SHORT {signal.symbol} qty={quantity} @ {fill_price:.2f} "
            f"commission={commission:.2f}"
        )
        return order

    def _close_position(
        self, signal: Signal, price: float, timestamp: datetime
    ) -> Order | None:
        """Close an existing position."""
        pos = self.positions.get(signal.symbol)
        if pos is None or not pos.is_open:
            logger.debug(f"No open position to close for {signal.symbol}")
            return None

        quantity = abs(pos.quantity)

        if pos.is_long:
            fill_price = price * (1 - self.slippage_rate)  # Sell at slightly lower
            proceeds = quantity * fill_price
            commission = proceeds * self.commission_rate
            self.cash += proceeds - commission
            pnl = (fill_price - pos.avg_entry_price) * quantity - commission
        else:
            fill_price = price * (1 + self.slippage_rate)  # Buy back at slightly higher
            cost = quantity * fill_price
            commission = cost * self.commission_rate
            self.cash -= cost + commission
            pnl = (pos.avg_entry_price - fill_price) * quantity - commission

        pnl_pct = pnl / (pos.avg_entry_price * quantity) if pos.avg_entry_price else 0.0

        # Record trade
        trade = TradeRecord(
            symbol=signal.symbol,
            direction="LONG" if pos.is_long else "SHORT",
            entry_time=pos.entry_time,
            entry_price=pos.avg_entry_price,
            exit_time=timestamp,
            exit_price=fill_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
        )
        self.trades.append(trade)

        # Clear position
        pos.quantity = 0.0

        order = Order(
            signal=signal,
            fill_price=fill_price,
            fill_quantity=quantity,
            commission=commission,
            fill_time=timestamp,
        )
        self.orders.append(order)
        logger.debug(
            f"CLOSE {signal.symbol} qty={quantity} @ {fill_price:.2f} "
            f"pnl={pnl:.2f} ({pnl_pct:.2%})"
        )
        return order
