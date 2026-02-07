"""Live/paper trading scheduler using APScheduler."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

from moneytalks.data.base import DataSource
from moneytalks.execution.paper import PaperTrader
from moneytalks.storage.database import Database
from moneytalks.strategy.base import Strategy
from moneytalks.strategy.signals import StrategyContext
from moneytalks.utils.logger import get_logger

logger = get_logger("execution.scheduler")


class TradingScheduler:
    """Scheduled strategy execution for paper/live trading.

    Periodically fetches new market data, runs the strategy's ``on_bar``,
    and submits any resulting signals through the broker.

    Example::

        scheduler = TradingScheduler(
            strategy=my_strategy,
            data_source=YFinanceSource(),
            broker=PaperTrader(),
            symbol="AAPL",
            interval="1m",
        )
        scheduler.start()  # Blocks until stopped
    """

    def __init__(
        self,
        strategy: Strategy,
        data_source: DataSource,
        broker: PaperTrader,
        symbol: str,
        interval: str = "1m",
        db: Database | None = None,
        lookback_bars: int = 100,
    ):
        self.strategy = strategy
        self.data_source = data_source
        self.broker = broker
        self.symbol = symbol
        self.interval = interval
        self.db = db
        self.lookback_bars = lookback_bars

        self._scheduler = BlockingScheduler()
        self._history: pd.DataFrame = pd.DataFrame()
        self._initialized = False

    def start(self):
        """Start the scheduled trading loop (blocking)."""
        # Map intervals to seconds for scheduling
        interval_seconds = _interval_to_seconds(self.interval)

        logger.info(
            f"Starting scheduler: {self.strategy.name} on {self.symbol} "
            f"every {interval_seconds}s"
        )

        # Run once immediately
        self._tick()

        # Schedule periodic execution
        self._scheduler.add_job(
            self._tick,
            "interval",
            seconds=interval_seconds,
            id="trading_tick",
        )

        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped by user")
            self.stop()

    def stop(self):
        """Stop the scheduler gracefully."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

    def _tick(self):
        """Single execution cycle: fetch data → run strategy → execute signals."""
        try:
            # Fetch latest data
            new_data = self.data_source.fetch_realtime(
                self.symbol, interval=self.interval
            )
            if new_data.empty:
                logger.warning(f"No data received for {self.symbol}")
                return

            # Merge with history
            if self._history.empty:
                self._history = new_data
            else:
                self._history = pd.concat([self._history, new_data])
                self._history = self._history[~self._history.index.duplicated(keep="last")]
                self._history = self._history.sort_index()

            # Keep only lookback window
            if len(self._history) > self.lookback_bars:
                self._history = self._history.iloc[-self.lookback_bars:]

            if self._history.empty:
                return

            # Build context
            context = StrategyContext(
                data=self._history,
                initial_capital=self.broker.initial_capital,
                current_bar_index=len(self._history) - 1,
                cash=self.broker.get_balance(),
                positions=dict(self.broker.positions),
            )

            # Initialize strategy once
            if not self._initialized:
                self.strategy.on_init(context)
                self._initialized = True

            # Run strategy on latest bar
            latest_bar = self._history.iloc[-1]
            self.broker.update_price(self.symbol, latest_bar["close"])
            signal = self.strategy.on_bar(latest_bar, context)

            # Execute signal
            if signal is not None:
                order = self.broker.submit_order(signal)
                logger.info(
                    f"Signal executed: {signal.direction.value} {signal.symbol} "
                    f"@ {order.fill_price:.2f} (reason: {signal.reason})"
                )

            # Save snapshot to database
            if self.db is not None:
                self.db.save_live_snapshot(
                    strategy_name=self.strategy.name,
                    portfolio_value=self.broker.get_portfolio_value(),
                    positions=self.broker.get_positions_dict(),
                )

            logger.debug(
                f"Tick complete | Portfolio: {self.broker.get_portfolio_value():,.2f} | "
                f"Cash: {self.broker.get_balance():,.2f}"
            )

        except Exception as e:
            logger.error(f"Error in tick: {e}", exc_info=True)


def _interval_to_seconds(interval: str) -> int:
    """Convert interval string to seconds for scheduling."""
    mapping = {
        "1m": 60,
        "2m": 120,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "60m": 3600,
        "90m": 5400,
        "1h": 3600,
        "1d": 86400,
    }
    result = mapping.get(interval)
    if result is None:
        raise ValueError(f"Cannot schedule for interval '{interval}'")
    return result
