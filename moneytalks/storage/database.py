"""SQLite database connection management and persistence helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from moneytalks.backtest.engine import BacktestResult
from moneytalks.backtest.metrics import MetricsReport
from moneytalks.backtest.portfolio import TradeRecord
from moneytalks.config import DB_PATH
from moneytalks.storage.models import BacktestRun, Base, LiveSnapshot, TradeModel
from moneytalks.utils.logger import get_logger

logger = get_logger("storage.database")


class Database:
    """SQLite database manager.

    Handles engine creation, table initialization, and provides
    convenience methods for saving/loading backtest results.
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized at {self.db_path}")

    def get_session(self) -> Session:
        """Create and return a new database session."""
        return self.SessionLocal()

    def save_backtest(
        self,
        result: BacktestResult,
        metrics: MetricsReport,
    ) -> int:
        """Save a backtest run and its trades to the database.

        Args:
            result: BacktestResult from the engine.
            metrics: Computed performance metrics.

        Returns:
            The ID of the saved BacktestRun record.
        """
        session = self.get_session()
        try:
            run = BacktestRun(
                strategy_name=result.strategy_name,
                params_json=json.dumps(result.params),
                symbol=result.symbol,
                interval=result.interval,
                start_date=result.data.index[0].to_pydatetime()
                if hasattr(result.data.index[0], "to_pydatetime")
                else result.data.index[0],
                end_date=result.data.index[-1].to_pydatetime()
                if hasattr(result.data.index[-1], "to_pydatetime")
                else result.data.index[-1],
                initial_capital=metrics.initial_capital,
                final_value=metrics.final_value,
                annual_return=metrics.annual_return,
                sharpe=metrics.sharpe_ratio,
                max_drawdown=metrics.max_drawdown,
                win_rate=metrics.win_rate,
                profit_loss_ratio=metrics.profit_loss_ratio,
                total_trades=metrics.total_trades,
                avg_holding_period=metrics.avg_holding_period,
                total_commission=metrics.total_commission,
            )
            session.add(run)
            session.flush()  # Get run.id

            # Save individual trades
            for trade in result.trades:
                trade_model = TradeModel(
                    backtest_run_id=run.id,
                    symbol=trade.symbol,
                    direction=trade.direction,
                    entry_time=trade.entry_time,
                    entry_price=trade.entry_price,
                    exit_time=trade.exit_time,
                    exit_price=trade.exit_price,
                    quantity=trade.quantity,
                    pnl=trade.pnl,
                    pnl_pct=trade.pnl_pct,
                    commission=trade.commission,
                )
                session.add(trade_model)

            session.commit()
            logger.info(
                f"Saved backtest run #{run.id}: {result.strategy_name} "
                f"({len(result.trades)} trades)"
            )
            return run.id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save backtest: {e}")
            raise
        finally:
            session.close()

    def save_live_snapshot(
        self,
        strategy_name: str,
        portfolio_value: float,
        positions: dict,
        pending_orders: list | None = None,
    ) -> int:
        """Save a live trading snapshot.

        Args:
            strategy_name: Name of the running strategy.
            portfolio_value: Current portfolio value.
            positions: Current positions dict.
            pending_orders: List of pending orders.

        Returns:
            The ID of the saved LiveSnapshot record.
        """
        session = self.get_session()
        try:
            snapshot = LiveSnapshot(
                strategy_name=strategy_name,
                portfolio_value=portfolio_value,
                positions_json=json.dumps(positions, default=str),
                pending_orders_json=json.dumps(pending_orders or [], default=str),
            )
            session.add(snapshot)
            session.commit()
            logger.debug(f"Saved live snapshot for {strategy_name}")
            return snapshot.id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save snapshot: {e}")
            raise
        finally:
            session.close()

    def list_backtest_runs(self, limit: int = 20) -> list[BacktestRun]:
        """List recent backtest runs.

        Args:
            limit: Maximum number of runs to return.

        Returns:
            List of BacktestRun records, most recent first.
        """
        session = self.get_session()
        try:
            runs = (
                session.query(BacktestRun)
                .order_by(BacktestRun.created_at.desc())
                .limit(limit)
                .all()
            )
            return runs
        finally:
            session.close()

    def get_backtest_trades(self, run_id: int) -> list[TradeModel]:
        """Get all trades for a specific backtest run.

        Args:
            run_id: BacktestRun ID.

        Returns:
            List of TradeModel records.
        """
        session = self.get_session()
        try:
            trades = (
                session.query(TradeModel)
                .filter(TradeModel.backtest_run_id == run_id)
                .all()
            )
            return trades
        finally:
            session.close()
