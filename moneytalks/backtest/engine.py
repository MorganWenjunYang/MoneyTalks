"""Event-driven backtesting engine."""

from __future__ import annotations

import pandas as pd

from moneytalks.backtest.portfolio import PortfolioManager
from moneytalks.config import (
    DEFAULT_COMMISSION_RATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_SLIPPAGE_RATE,
)
from moneytalks.strategy.base import Strategy
from moneytalks.strategy.signals import StrategyContext
from moneytalks.utils.logger import get_logger

logger = get_logger("backtest.engine")


class BacktestResult:
    """Container for backtest outputs."""

    def __init__(
        self,
        strategy_name: str,
        params: dict,
        symbol: str,
        interval: str,
        data: pd.DataFrame,
        portfolio: PortfolioManager,
    ):
        self.strategy_name = strategy_name
        self.params = params
        self.symbol = symbol
        self.interval = interval
        self.data = data
        self.portfolio = portfolio

    @property
    def equity_series(self) -> pd.Series:
        return self.portfolio.get_equity_series()

    @property
    def trades(self):
        return self.portfolio.trades

    @property
    def initial_capital(self) -> float:
        return self.portfolio.initial_capital

    @property
    def final_value(self) -> float:
        equity = self.equity_series
        if equity.empty:
            return self.portfolio.cash
        return equity.iloc[-1]


class BacktestEngine:
    """Event-driven backtesting engine.

    Iterates through historical data bar-by-bar, calling the strategy's
    ``on_bar`` method and executing any resulting signals through the
    PortfolioManager.

    Example::

        engine = BacktestEngine()
        result = engine.run(
            strategy=SMACrossStrategy({"fast_period": 10, "slow_period": 30}),
            data=ohlcv_dataframe,
            symbol="AAPL",
        )
    """

    def __init__(
        self,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        interval: str = "1d",
    ) -> BacktestResult:
        """Run a backtest.

        Args:
            strategy: The strategy instance to test.
            data: OHLCV DataFrame with DatetimeIndex.
            symbol: Ticker symbol (for labeling).
            interval: Data interval (for labeling).

        Returns:
            BacktestResult containing portfolio state, trades, and equity curve.
        """
        if data.empty:
            raise ValueError("Cannot backtest on empty data")

        logger.info(
            f"Starting backtest: {strategy.name} on {symbol} "
            f"({len(data)} bars, {data.index[0]} -> {data.index[-1]})"
        )

        portfolio = PortfolioManager(
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate,
        )

        context = StrategyContext(
            data=data,
            initial_capital=self.initial_capital,
        )

        # Initialize strategy
        strategy.on_init(context)

        # Main event loop
        for i in range(len(data)):
            bar = data.iloc[i]

            # Update context
            context.update(
                bar_index=i,
                cash=portfolio.cash,
                positions=dict(portfolio.positions),
            )

            # Get signal from strategy
            signal = strategy.on_bar(bar, context)

            # Execute signal
            if signal is not None:
                portfolio.execute(signal, bar)

            # Record equity
            price_map = {symbol: bar["close"]}
            for sym in portfolio.positions:
                if sym != symbol:
                    price_map[sym] = bar.get("close", 0.0)
            portfolio.record_equity(bar.name, price_map)

        # Close any remaining open positions at last bar
        self._close_all_positions(portfolio, data.iloc[-1])

        logger.info(
            f"Backtest complete: {strategy.name} | "
            f"Final value: {portfolio.get_equity_series().iloc[-1]:,.2f} | "
            f"Trades: {len(portfolio.trades)}"
        )

        return BacktestResult(
            strategy_name=strategy.name,
            params=strategy.params,
            symbol=symbol,
            interval=interval,
            data=data,
            portfolio=portfolio,
        )

    @staticmethod
    def _close_all_positions(portfolio: PortfolioManager, last_bar: pd.Series):
        """Force-close all open positions at the end of backtest."""
        from moneytalks.strategy.signals import Signal
        from moneytalks.utils.types import Direction, OrderType

        for symbol, pos in list(portfolio.positions.items()):
            if pos.is_open:
                signal = Signal(
                    direction=Direction.CLOSE,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    reason="End of backtest: force close",
                )
                portfolio.execute(signal, last_bar)
