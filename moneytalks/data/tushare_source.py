"""Tushare data source implementation for A-share (Chinese stock market)."""

from __future__ import annotations

import pandas as pd
import tushare as ts

from moneytalks.config import TUSHARE_TOKEN
from moneytalks.data.base import DataSource
from moneytalks.utils.logger import get_logger

logger = get_logger("data.tushare")

# Standard column names used throughout the system
STANDARD_COLUMNS = ["open", "high", "low", "close", "volume"]


class TushareSource(DataSource):
    """Data source backed by Tushare Pro API (A-share market).

    Provides access to Chinese A-share daily market data via the
    Tushare Pro platform.  A valid token is required — set it in
    config.py or via the ``TUSHARE_TOKEN`` environment variable.

    Supported intervals:
        Daily: 1d  (primary, via ``pro.daily``)
        Weekly: 1wk (via ``pro.weekly``)
        Monthly: 1mo (via ``pro.monthly``)

    Symbol format:
        Tushare uses ``ts_code`` like ``000001.SZ``, ``600000.SH``.
        This source accepts both formats:
        - Full:  ``000001.SZ``
        - Short: ``000001`` (auto-detects SZ/SH suffix)

    Notes:
        - Date format is converted internally (YYYY-MM-DD → YYYYMMDD).
        - Volume is in *shares* (Tushare returns 手, multiplied by 100).
        - Timestamps are localized to Asia/Shanghai then converted to UTC.
    """

    _SUPPORTED_INTERVALS = ["1d", "1wk", "1mo"]

    def __init__(self, token: str | None = None):
        """Initialize with a Tushare Pro token.

        Args:
            token: Tushare Pro API token.  Falls back to config / env var.
        """
        self._token = token or TUSHARE_TOKEN
        if not self._token:
            logger.warning(
                "No Tushare token configured. Set TUSHARE_TOKEN env var "
                "or pass token= to TushareSource()."
            )
        ts.set_token(self._token)
        self._pro = ts.pro_api()

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for an A-share symbol.

        Args:
            symbol: Tushare ts_code (e.g. "000001.SZ") or short code ("000001").
            start: Start date, e.g. "2023-01-01".
            end: End date, e.g. "2024-01-01".
            interval: "1d", "1wk", or "1mo".

        Returns:
            Standardised DataFrame with DatetimeIndex (UTC).
        """
        self._validate_interval(interval)
        ts_code = self._normalize_symbol(symbol)
        start_date = self._to_tushare_date(start)
        end_date = self._to_tushare_date(end)

        logger.info(
            f"Fetching historical data: {ts_code} [{start_date} -> {end_date}] "
            f"interval={interval}"
        )

        try:
            if interval == "1d":
                df = self._pro.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                )
            elif interval == "1wk":
                df = self._pro.weekly(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                )
            elif interval == "1mo":
                df = self._pro.monthly(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Tushare API error: {e}")
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        if df is None or df.empty:
            logger.warning(f"No data returned for {ts_code}")
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        return self._standardize(df)

    def fetch_realtime(self, symbol: str, interval: str = "1m") -> pd.DataFrame:
        """Fetch recent data.  For Tushare daily, returns today's data.

        Note: Tushare free tier does not support real-time minute data.
        This falls back to fetching the most recent trading day's daily bar.
        """
        ts_code = self._normalize_symbol(symbol)
        logger.info(f"Fetching latest daily bar for {ts_code}")

        try:
            df = self._pro.daily(ts_code=ts_code, start_date="", end_date="")
            if df is not None and not df.empty:
                df = df.head(1)  # Most recent row
                return self._standardize(df)
        except Exception as e:
            logger.error(f"Tushare realtime error: {e}")

        return pd.DataFrame(columns=STANDARD_COLUMNS)

    def supported_intervals(self) -> list[str]:
        return list(self._SUPPORTED_INTERVALS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_interval(self, interval: str) -> None:
        if interval not in self._SUPPORTED_INTERVALS:
            raise ValueError(
                f"Unsupported interval '{interval}' for Tushare. "
                f"Supported: {self._SUPPORTED_INTERVALS}"
            )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize symbol to Tushare ts_code format (e.g. 000001.SZ).

        Rules:
            - If already contains '.', return as-is (upper-cased).
            - 6-digit codes starting with 6 → append .SH (Shanghai).
            - Other 6-digit codes → append .SZ (Shenzhen).
            - Codes starting with 4/8 → append .BJ (Beijing / 北交所).
        """
        symbol = symbol.strip().upper()
        if "." in symbol:
            return symbol

        code = symbol.lstrip("0") if len(symbol) > 6 else symbol
        # Pad to 6 digits
        code = symbol.zfill(6)

        if code.startswith("6"):
            return f"{code}.SH"
        elif code.startswith(("4", "8")):
            return f"{code}.BJ"
        else:
            return f"{code}.SZ"

    @staticmethod
    def _to_tushare_date(date_str: str) -> str:
        """Convert 'YYYY-MM-DD' or 'YYYYMMDD' to Tushare's YYYYMMDD format."""
        return date_str.replace("-", "").replace("/", "")

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize Tushare output to system convention.

        Tushare daily returns columns:
            ts_code, trade_date, open, high, low, close, pre_close,
            change, pct_chg, vol (手), amount (千元)
        """
        if df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        # Build trade_date index
        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.set_index("trade_date")
        df = df.sort_index()
        df.index.name = "timestamp"

        # Rename and select standard columns
        rename_map = {"vol": "volume"}
        df = df.rename(columns=rename_map)

        # Tushare vol is in 手 (lots of 100 shares) — convert to shares
        if "volume" in df.columns:
            df["volume"] = df["volume"] * 100

        # Keep only standard columns
        available = [c for c in STANDARD_COLUMNS if c in df.columns]
        df = df[available]

        # Localize to Shanghai time, then convert to UTC
        df.index = df.index.tz_localize("Asia/Shanghai").tz_convert("UTC")

        return df
