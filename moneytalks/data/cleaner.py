"""Data cleaning utilities: deduplication, gap filling, time alignment."""

from __future__ import annotations

import pandas as pd

from moneytalks.utils.logger import get_logger

logger = get_logger("data.cleaner")


class DataCleaner:
    """Pipeline for cleaning raw OHLCV DataFrames.

    Operations:
        1. deduplicate  – remove duplicate timestamps, keep last
        2. fill_gaps    – forward-fill missing bars, mark filled rows
        3. align_time   – snap timestamps to a regular time grid in UTC
    """

    def clean(self, df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
        """Run the full cleaning pipeline.

        Args:
            df: Raw OHLCV DataFrame with DatetimeIndex named 'timestamp'.
            interval: Expected bar interval (used for gap detection).

        Returns:
            Cleaned DataFrame with an additional boolean column 'filled'
            indicating rows that were forward-filled.
        """
        if df.empty:
            return df

        df = self.deduplicate(df)
        df = self.align_time(df, interval)
        df = self.fill_gaps(df)
        return df

    # ------------------------------------------------------------------
    # Individual cleaning steps
    # ------------------------------------------------------------------

    @staticmethod
    def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps, keeping the last occurrence."""
        before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        removed = before - len(df)
        if removed > 0:
            logger.info(f"Deduplicated: removed {removed} duplicate rows")
        return df

    @staticmethod
    def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill missing values and mark filled rows.

        Adds a boolean column 'filled' — True for rows that were NaN
        before filling.
        """
        # Identify rows that are entirely NaN (i.e., gaps created by reindex)
        is_gap = df[["open", "high", "low", "close"]].isna().all(axis=1)
        df = df.ffill()

        # Drop any remaining NaN rows at the beginning (no prior data to fill)
        df = df.dropna(subset=["close"])

        df["filled"] = False
        df.loc[is_gap & df.index.isin(df.index), "filled"] = True

        filled_count = df["filled"].sum()
        if filled_count > 0:
            logger.info(f"Filled {filled_count} gap rows via forward-fill")
        return df

    @staticmethod
    def align_time(df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
        """Align timestamps to a regular time grid in UTC.

        Args:
            df: DataFrame with DatetimeIndex.
            interval: Bar interval string (e.g., "1m", "1d").

        Returns:
            DataFrame reindexed to a regular frequency grid.
        """
        freq = _interval_to_freq(interval)
        if freq is None:
            # Cannot create a regular grid for this interval; just sort
            return df.sort_index()

        # Ensure UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif str(df.index.tz) != "UTC":
            df.index = df.index.tz_convert("UTC")

        # Build a regular time grid from first to last timestamp
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq,
            tz="UTC",
            name="timestamp",
        )

        df = df.reindex(full_index)
        return df


def _interval_to_freq(interval: str) -> str | None:
    """Map an interval string to a pandas frequency alias."""
    mapping = {
        "1m": "1min",
        "2m": "2min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "90m": "90min",
        "1h": "1h",
        "1d": "1D",
        "5d": "5D",
        "1wk": "1W",
        "1mo": "1ME",
    }
    return mapping.get(interval)
