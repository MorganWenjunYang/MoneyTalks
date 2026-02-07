"""Parquet-based local data store for OHLCV market data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from moneytalks.config import DATA_DIR
from moneytalks.utils.logger import get_logger

logger = get_logger("data.store")


class ParquetStore:
    """Local cache for OHLCV data using Parquet files.

    Directory layout::

        data/
        ├── AAPL/
        │   ├── 1d.parquet
        │   └── 1m.parquet
        └── MSFT/
            └── 1d.parquet
    """

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir) if base_dir else DATA_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _parquet_path(self, symbol: str, interval: str) -> Path:
        """Return the Parquet file path for a given symbol and interval."""
        symbol_dir = self.base_dir / symbol.upper()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{interval}.parquet"

    def has_data(self, symbol: str, interval: str) -> bool:
        """Check if cached data exists for the given symbol/interval."""
        return self._parquet_path(symbol, interval).exists()

    def load(self, symbol: str, interval: str) -> pd.DataFrame:
        """Load cached data from Parquet.

        Returns:
            DataFrame with DatetimeIndex, or empty DataFrame if no cache.
        """
        path = self._parquet_path(symbol, interval)
        if not path.exists():
            logger.debug(f"No cache found: {path}")
            return pd.DataFrame()

        df = pd.read_parquet(path)

        # Restore DatetimeIndex
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index.name = "timestamp"

        logger.debug(f"Loaded {len(df)} rows from {path}")
        return df

    def save(self, df: pd.DataFrame, symbol: str, interval: str) -> None:
        """Save data to Parquet, merging with existing cache.

        If cached data exists, the new data is merged (new rows override
        existing ones on the same timestamp).

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
            symbol: Ticker symbol.
            interval: Data interval.
        """
        if df.empty:
            return

        path = self._parquet_path(symbol, interval)

        # Merge with existing cache
        if path.exists():
            existing = self.load(symbol, interval)
            if not existing.empty:
                df = pd.concat([existing, df])
                df = df[~df.index.duplicated(keep="last")]
                df = df.sort_index()

        # Write index as a column so it round-trips correctly
        out = df.copy()
        out.index.name = "timestamp"
        out = out.reset_index()
        out.to_parquet(path, index=False, engine="pyarrow")
        logger.info(f"Saved {len(out)} rows to {path}")

    def get_date_range(
        self, symbol: str, interval: str
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Return (min_date, max_date) of cached data, or (None, None)."""
        df = self.load(symbol, interval)
        if df.empty:
            return None, None
        return df.index.min(), df.index.max()

    def delete(self, symbol: str, interval: str) -> None:
        """Delete cached data for a symbol/interval."""
        path = self._parquet_path(symbol, interval)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted cache: {path}")
