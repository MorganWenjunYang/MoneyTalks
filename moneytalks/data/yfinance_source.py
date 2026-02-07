"""YFinance data source implementation."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from moneytalks.data.base import DataSource
from moneytalks.utils.logger import get_logger

logger = get_logger("data.yfinance")

# Standard column names used throughout the system
STANDARD_COLUMNS = ["open", "high", "low", "close", "volume"]


class YFinanceSource(DataSource):
    """Data source backed by the yfinance library (US stocks + global markets).

    Supported intervals:
        Intraday (max 30 days history): 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h
        Daily and above: 1d, 5d, 1wk, 1mo

    Notes:
        - Intraday data is only available for the most recent 30 days.
        - All timestamps are converted to UTC.
    """

    _SUPPORTED_INTERVALS = [
        "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
        "1d", "5d", "1wk", "1mo",
    ]

    def fetch_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data via yfinance."""
        self._validate_interval(interval)
        logger.info(
            f"Fetching historical data: {symbol} [{start} -> {end}] interval={interval}"
        )

        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        return self._standardize(df)

    def fetch_realtime(self, symbol: str, interval: str = "1m") -> pd.DataFrame:
        """Fetch recent intraday data via yfinance Ticker."""
        self._validate_interval(interval)
        logger.info(f"Fetching realtime data: {symbol} interval={interval}")

        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval=interval)

        if df.empty:
            logger.warning(f"No realtime data returned for {symbol}")
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        return self._standardize(df)

    def supported_intervals(self) -> list[str]:
        """Return supported interval strings."""
        return list(self._SUPPORTED_INTERVALS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_interval(self, interval: str) -> None:
        if interval not in self._SUPPORTED_INTERVALS:
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Supported: {self._SUPPORTED_INTERVALS}"
            )

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and index to the system convention."""
        # yfinance may return MultiIndex columns for multi-ticker downloads
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(level=1, axis=1)

        # Lowercase all column names
        df.columns = [c.lower() for c in df.columns]

        # Keep only standard columns
        available = [c for c in STANDARD_COLUMNS if c in df.columns]
        df = df[available].copy()

        # Ensure index is a DatetimeIndex named 'timestamp'
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index.name = "timestamp"

        # Convert to UTC if timezone-aware, otherwise localize to UTC
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC")
        else:
            df.index = df.index.tz_localize("UTC")

        return df
