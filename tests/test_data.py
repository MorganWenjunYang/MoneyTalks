"""Tests for the data layer: cleaner and store."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moneytalks.data.cleaner import DataCleaner
from moneytalks.data.store import ParquetStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create a sample OHLCV DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=50, freq="1D", tz="UTC")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(50) * 2)
    df = pd.DataFrame(
        {
            "open": close - np.random.rand(50),
            "high": close + np.abs(np.random.randn(50)),
            "low": close - np.abs(np.random.randn(50)),
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, size=50),
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def tmp_store(tmp_path) -> ParquetStore:
    """Create a ParquetStore backed by a temporary directory."""
    return ParquetStore(base_dir=tmp_path)


# ---------------------------------------------------------------------------
# DataCleaner tests
# ---------------------------------------------------------------------------


class TestDataCleaner:
    def test_deduplicate(self, sample_ohlcv: pd.DataFrame):
        # Add a duplicate row
        dup = sample_ohlcv.iloc[[0]]
        df = pd.concat([sample_ohlcv, dup])
        assert len(df) == 51

        cleaner = DataCleaner()
        result = cleaner.deduplicate(df)
        assert len(result) == 50

    def test_fill_gaps(self, sample_ohlcv: pd.DataFrame):
        # Remove some rows to create gaps
        df = sample_ohlcv.drop(sample_ohlcv.index[5:8])  # Remove 3 rows
        assert len(df) == 47

        cleaner = DataCleaner()
        aligned = cleaner.align_time(df, "1d")
        result = cleaner.fill_gaps(aligned)

        # Should have 50 rows again
        assert len(result) == 50
        assert "filled" in result.columns
        assert result["filled"].sum() == 3

    def test_clean_pipeline(self, sample_ohlcv: pd.DataFrame):
        cleaner = DataCleaner()
        result = cleaner.clean(sample_ohlcv, "1d")

        assert not result.empty
        assert "filled" in result.columns
        assert result.index.is_monotonic_increasing

    def test_clean_empty(self):
        cleaner = DataCleaner()
        result = cleaner.clean(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# ParquetStore tests
# ---------------------------------------------------------------------------


class TestParquetStore:
    def test_save_and_load(self, tmp_store: ParquetStore, sample_ohlcv: pd.DataFrame):
        tmp_store.save(sample_ohlcv, "AAPL", "1d")
        assert tmp_store.has_data("AAPL", "1d")

        loaded = tmp_store.load("AAPL", "1d")
        assert len(loaded) == len(sample_ohlcv)
        assert list(loaded.columns) == list(sample_ohlcv.columns)

    def test_incremental_update(
        self, tmp_store: ParquetStore, sample_ohlcv: pd.DataFrame
    ):
        # Save first 30 bars
        tmp_store.save(sample_ohlcv.iloc[:30], "AAPL", "1d")
        assert tmp_store.has_data("AAPL", "1d")

        # Save last 30 bars (10 overlap)
        tmp_store.save(sample_ohlcv.iloc[20:], "AAPL", "1d")

        loaded = tmp_store.load("AAPL", "1d")
        assert len(loaded) == 50  # No duplicates

    def test_load_missing(self, tmp_store: ParquetStore):
        result = tmp_store.load("NONEXISTENT", "1d")
        assert result.empty

    def test_has_data_missing(self, tmp_store: ParquetStore):
        assert not tmp_store.has_data("NONEXISTENT", "1d")

    def test_get_date_range(
        self, tmp_store: ParquetStore, sample_ohlcv: pd.DataFrame
    ):
        tmp_store.save(sample_ohlcv, "AAPL", "1d")
        min_dt, max_dt = tmp_store.get_date_range("AAPL", "1d")
        assert min_dt is not None
        assert max_dt is not None
        assert min_dt <= max_dt

    def test_delete(self, tmp_store: ParquetStore, sample_ohlcv: pd.DataFrame):
        tmp_store.save(sample_ohlcv, "AAPL", "1d")
        assert tmp_store.has_data("AAPL", "1d")

        tmp_store.delete("AAPL", "1d")
        assert not tmp_store.has_data("AAPL", "1d")
