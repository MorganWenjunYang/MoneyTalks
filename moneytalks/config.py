"""Global configuration for MoneyTalks."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local data cache directory (Parquet files)
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# SQLite database path
DB_PATH = PROJECT_ROOT / "moneytalks.db"

# Default backtest settings
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_COMMISSION_RATE = 0.001  # 0.1%
DEFAULT_SLIPPAGE_RATE = 0.0005  # 0.05%

# Default data settings
DEFAULT_INTERVAL = "1d"
DEFAULT_TIMEZONE = "UTC"

# Annualization factor (trading days per year)
TRADING_DAYS_PER_YEAR = 252

# Tushare configuration
# Set your token here or via environment variable TUSHARE_TOKEN
import os
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
