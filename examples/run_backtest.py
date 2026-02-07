#!/usr/bin/env python3
"""Example: Run a SMA crossover backtest on AAPL.

Usage:
    python examples/run_backtest.py
"""

from moneytalks.backtest.engine import BacktestEngine
from moneytalks.backtest.report import ReportGenerator
from moneytalks.data.cleaner import DataCleaner
from moneytalks.data.store import ParquetStore
from moneytalks.data.yfinance_source import YFinanceSource
from moneytalks.storage.database import Database
from moneytalks.strategy.examples.sma_cross import SMACrossStrategy


def main():
    # --- Configuration ---
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2024-12-31"
    interval = "1d"
    initial_capital = 100_000.0

    # --- Step 1: Fetch data ---
    print(f"Fetching {symbol} data from {start_date} to {end_date}...")
    source = YFinanceSource()
    store = ParquetStore()
    cleaner = DataCleaner()

    # Try loading from cache first
    data = store.load(symbol, interval)
    if data.empty:
        data = source.fetch_historical(symbol, start_date, end_date, interval)
        data = cleaner.clean(data, interval)
        store.save(data, symbol, interval)
        print(f"Fetched and cached {len(data)} bars")
    else:
        print(f"Loaded {len(data)} bars from cache")

    # Remove the 'filled' column if present (from cleaner)
    if "filled" in data.columns:
        data = data.drop(columns=["filled"])

    print(f"Data range: {data.index[0]} -> {data.index[-1]}")
    print(f"Total bars: {len(data)}")
    print()

    # --- Step 2: Create strategy ---
    strategy = SMACrossStrategy(params={
        "fast_period": 10,
        "slow_period": 30,
        "symbol": symbol,
    })

    # --- Step 3: Run backtest ---
    engine = BacktestEngine(initial_capital=initial_capital)
    result = engine.run(
        strategy=strategy,
        data=data,
        symbol=symbol,
        interval=interval,
    )

    # --- Step 4: Generate report ---
    reporter = ReportGenerator()
    metrics = reporter.generate(result, save_charts=True, chart_dir="output")

    # --- Step 5: Save to database ---
    db = Database()
    run_id = db.save_backtest(result, metrics)
    print(f"\nBacktest saved to database (run_id={run_id})")

    # --- Step 6: Print trade log ---
    print(f"\n{'='*70}")
    print(f"  TRADE LOG ({len(result.trades)} trades)")
    print(f"{'='*70}")
    for i, trade in enumerate(result.trades, 1):
        print(
            f"  #{i:3d} | {trade.direction:5s} | "
            f"Entry: {trade.entry_price:8.2f} @ {trade.entry_time} | "
            f"Exit: {trade.exit_price:8.2f} @ {trade.exit_time} | "
            f"PnL: {trade.pnl:+10.2f} ({trade.pnl_pct:+.2%})"
        )


if __name__ == "__main__":
    main()
