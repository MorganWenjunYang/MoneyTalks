"""SQLAlchemy ORM models for persisting backtest results and trade records."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class BacktestRun(Base):
    """A single backtest execution record."""

    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(128), nullable=False)
    params_json = Column(Text, default="{}")
    symbol = Column(String(32), nullable=False)
    interval = Column(String(16), nullable=False)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_capital = Column(Float, nullable=False)
    final_value = Column(Float)
    annual_return = Column(Float)
    sharpe = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_loss_ratio = Column(Float)
    total_trades = Column(Integer)
    avg_holding_period = Column(Float)
    total_commission = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to trades
    trades = relationship("TradeModel", back_populates="backtest_run", cascade="all, delete-orphan")

    def __repr__(self):
        return (
            f"<BacktestRun(id={self.id}, strategy={self.strategy_name}, "
            f"symbol={self.symbol}, return={self.annual_return:.2%})>"
        )


class TradeModel(Base):
    """A single trade (round-trip: entry + exit)."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False)
    symbol = Column(String(32), nullable=False)
    direction = Column(String(8), nullable=False)
    entry_time = Column(DateTime)
    entry_price = Column(Float)
    exit_time = Column(DateTime)
    exit_price = Column(Float)
    quantity = Column(Float)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    commission = Column(Float)

    # Relationship back to backtest run
    backtest_run = relationship("BacktestRun", back_populates="trades")

    def __repr__(self):
        return (
            f"<Trade(id={self.id}, symbol={self.symbol}, "
            f"dir={self.direction}, pnl={self.pnl:.2f})>"
        )


class LiveSnapshot(Base):
    """A point-in-time snapshot of a live/paper trading session."""

    __tablename__ = "live_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(128), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    portfolio_value = Column(Float)
    positions_json = Column(Text, default="{}")
    pending_orders_json = Column(Text, default="[]")

    def __repr__(self):
        return (
            f"<LiveSnapshot(id={self.id}, strategy={self.strategy_name}, "
            f"value={self.portfolio_value:.2f})>"
        )
