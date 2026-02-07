"""Abstract base class for data sources."""

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Abstract interface for financial data providers.

    All data sources must implement this interface, enabling seamless
    switching between providers (yfinance, CCXT, AKShare, etc.).
    """

    @abstractmethod
    def fetch_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol (e.g., "AAPL", "MSFT").
            start: Start date string (e.g., "2023-01-01").
            end: End date string (e.g., "2024-01-01").
            interval: Data granularity (e.g., "1m", "1d").

        Returns:
            DataFrame with columns: open, high, low, close, volume.
            Index is a DatetimeIndex in UTC.
        """
        ...

    @abstractmethod
    def fetch_realtime(self, symbol: str, interval: str = "1m") -> pd.DataFrame:
        """Fetch recent real-time / intraday data for a symbol.

        Args:
            symbol: Ticker symbol.
            interval: Intraday granularity (e.g., "1m", "5m").

        Returns:
            DataFrame with columns: open, high, low, close, volume.
            Index is a DatetimeIndex in UTC.
        """
        ...

    @abstractmethod
    def supported_intervals(self) -> list[str]:
        """Return the list of supported interval strings."""
        ...
