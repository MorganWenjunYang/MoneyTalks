"""Paper trading (simulated) broker implementation."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from moneytalks.config import (
    DEFAULT_COMMISSION_RATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_SLIPPAGE_RATE,
)
from moneytalks.execution.base import Broker
from moneytalks.strategy.signals import Order, Position, Signal
from moneytalks.utils.logger import get_logger
from moneytalks.utils.types import Direction

logger = get_logger("execution.paper")


class PaperTrader(Broker):
    """Simulated broker for paper trading.

    Uses real-time market data to simulate order execution, tracking
    positions, cash, and trade history without risking real money.
    """

    def __init__(
        self,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self._last_prices: dict[str, float] = {}

    def update_price(self, symbol: str, price: float):
        """Update the latest known price for a symbol.

        This should be called with real-time data before submitting orders.
        """
        self._last_prices[symbol] = price

    def submit_order(self, signal: Signal) -> Order:
        """Execute a paper trade based on the signal and last known price."""
        price = signal.price or self._last_prices.get(signal.symbol)
        if price is None:
            raise ValueError(
                f"No price available for {signal.symbol}. "
                "Call update_price() first or set signal.price."
            )

        now = datetime.utcnow()

        if signal.direction == Direction.LONG:
            return self._execute_long(signal, price, now)
        elif signal.direction == Direction.SHORT:
            return self._execute_short(signal, price, now)
        elif signal.direction == Direction.CLOSE:
            return self._execute_close(signal, price, now)
        else:
            raise ValueError(f"Unknown direction: {signal.direction}")

    def get_position(self, symbol: str) -> Position | None:
        pos = self.positions.get(symbol)
        if pos and pos.is_open:
            return pos
        return None

    def get_balance(self) -> float:
        return self.cash

    def get_portfolio_value(self) -> float:
        value = self.cash
        for symbol, pos in self.positions.items():
            if pos.is_open:
                price = self._last_prices.get(symbol, pos.avg_entry_price)
                value += pos.quantity * price
        return value

    def get_positions_dict(self) -> dict:
        """Return positions as a serializable dict."""
        result = {}
        for symbol, pos in self.positions.items():
            if pos.is_open:
                result[symbol] = {
                    "quantity": pos.quantity,
                    "avg_entry_price": pos.avg_entry_price,
                    "entry_time": str(pos.entry_time) if pos.entry_time else None,
                }
        return result

    # ------------------------------------------------------------------
    # Private execution helpers
    # ------------------------------------------------------------------

    def _execute_long(self, signal: Signal, price: float, now: datetime) -> Order:
        fill_price = price * (1 + self.slippage_rate)
        quantity = signal.quantity
        if quantity <= 0:
            quantity = int(self.cash * 0.95 / fill_price)
        if quantity <= 0:
            logger.warning(f"Insufficient cash for LONG {signal.symbol}")
            return Order(signal=signal)

        cost = quantity * fill_price
        commission = cost * self.commission_rate
        if cost + commission > self.cash:
            quantity = int(self.cash * 0.95 / (fill_price * (1 + self.commission_rate)))
            if quantity <= 0:
                return Order(signal=signal)
            cost = quantity * fill_price
            commission = cost * self.commission_rate

        self.cash -= cost + commission
        self.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            quantity=quantity,
            avg_entry_price=fill_price,
            entry_time=now,
        )

        order = Order(
            signal=signal,
            fill_price=fill_price,
            fill_quantity=quantity,
            commission=commission,
            fill_time=now,
        )
        self.orders.append(order)
        logger.info(f"PAPER LONG {signal.symbol} qty={quantity} @ {fill_price:.2f}")
        return order

    def _execute_short(self, signal: Signal, price: float, now: datetime) -> Order:
        fill_price = price * (1 - self.slippage_rate)
        quantity = signal.quantity
        if quantity <= 0:
            quantity = int(self.cash * 0.95 / fill_price)
        if quantity <= 0:
            logger.warning(f"Insufficient margin for SHORT {signal.symbol}")
            return Order(signal=signal)

        proceeds = quantity * fill_price
        commission = proceeds * self.commission_rate
        self.cash += proceeds - commission

        self.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            quantity=-quantity,
            avg_entry_price=fill_price,
            entry_time=now,
        )

        order = Order(
            signal=signal,
            fill_price=fill_price,
            fill_quantity=quantity,
            commission=commission,
            fill_time=now,
        )
        self.orders.append(order)
        logger.info(f"PAPER SHORT {signal.symbol} qty={quantity} @ {fill_price:.2f}")
        return order

    def _execute_close(self, signal: Signal, price: float, now: datetime) -> Order:
        pos = self.positions.get(signal.symbol)
        if pos is None or not pos.is_open:
            logger.debug(f"No position to close for {signal.symbol}")
            return Order(signal=signal)

        quantity = abs(pos.quantity)
        if pos.is_long:
            fill_price = price * (1 - self.slippage_rate)
            proceeds = quantity * fill_price
            commission = proceeds * self.commission_rate
            self.cash += proceeds - commission
        else:
            fill_price = price * (1 + self.slippage_rate)
            cost = quantity * fill_price
            commission = cost * self.commission_rate
            self.cash -= cost + commission

        pos.quantity = 0.0

        order = Order(
            signal=signal,
            fill_price=fill_price,
            fill_quantity=quantity,
            commission=commission,
            fill_time=now,
        )
        self.orders.append(order)
        logger.info(f"PAPER CLOSE {signal.symbol} qty={quantity} @ {fill_price:.2f}")
        return order
