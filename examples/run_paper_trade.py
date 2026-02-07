#!/usr/bin/env python3
"""Example: Run paper trading with SMA crossover strategy on AAPL.

This script starts a simulated trading session that periodically
fetches real-time data and executes the strategy.

Usage:
    python examples/run_paper_trade.py

Press Ctrl+C to stop.
"""

from moneytalks.data.yfinance_source import YFinanceSource
from moneytalks.execution.paper import PaperTrader
from moneytalks.execution.scheduler import TradingScheduler
from moneytalks.storage.database import Database
from moneytalks.strategy.examples.sma_cross import SMACrossStrategy


def main():
    # --- Configuration ---
    symbol = "AAPL"
    interval = "1m"  # 1-minute bars
    initial_capital = 100_000.0

    print(f"Starting paper trading: SMA Cross on {symbol}")
    print(f"Interval: {interval} | Capital: ${initial_capital:,.2f}")
    print("Press Ctrl+C to stop.\n")

    # --- Setup components ---
    strategy = SMACrossStrategy(params={
        "fast_period": 5,
        "slow_period": 15,
        "symbol": symbol,
    })

    data_source = YFinanceSource()
    broker = PaperTrader(initial_capital=initial_capital)
    db = Database()

    # --- Start scheduler ---
    scheduler = TradingScheduler(
        strategy=strategy,
        data_source=data_source,
        broker=broker,
        symbol=symbol,
        interval=interval,
        db=db,
        lookback_bars=100,
    )

    scheduler.start()

    # Print final state after stop
    print(f"\nFinal portfolio value: ${broker.get_portfolio_value():,.2f}")
    print(f"Cash: ${broker.get_balance():,.2f}")
    print(f"Total orders: {len(broker.orders)}")


if __name__ == "__main__":
    main()
