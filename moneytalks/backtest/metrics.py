"""Performance metrics calculation for backtesting results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from moneytalks.backtest.portfolio import TradeRecord
from moneytalks.config import TRADING_DAYS_PER_YEAR


@dataclass
class MetricsReport:
    """Container for all calculated performance metrics."""

    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in bars
    win_rate: float
    profit_loss_ratio: float
    total_trades: int
    avg_holding_period: float  # in bars/days
    total_return: float
    total_commission: float
    initial_capital: float
    final_value: float

    def to_dict(self) -> dict:
        """Convert metrics to a dictionary."""
        return {
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "win_rate": self.win_rate,
            "profit_loss_ratio": self.profit_loss_ratio,
            "total_trades": self.total_trades,
            "avg_holding_period": self.avg_holding_period,
            "total_return": self.total_return,
            "total_commission": self.total_commission,
            "initial_capital": self.initial_capital,
            "final_value": self.final_value,
        }

    def summary(self) -> str:
        """Return a formatted text summary of the metrics."""
        lines = [
            "=" * 50,
            "          BACKTEST PERFORMANCE REPORT",
            "=" * 50,
            f"  Initial Capital:      {self.initial_capital:>14,.2f}",
            f"  Final Value:          {self.final_value:>14,.2f}",
            f"  Total Return:         {self.total_return:>13.2%}",
            f"  Annual Return:        {self.annual_return:>13.2%}",
            f"  Sharpe Ratio:         {self.sharpe_ratio:>14.3f}",
            f"  Max Drawdown:         {self.max_drawdown:>13.2%}",
            f"  Max DD Duration:      {self.max_drawdown_duration:>11d} bars",
            "-" * 50,
            f"  Total Trades:         {self.total_trades:>14d}",
            f"  Win Rate:             {self.win_rate:>13.2%}",
            f"  Profit/Loss Ratio:    {self.profit_loss_ratio:>14.3f}",
            f"  Avg Holding Period:   {self.avg_holding_period:>11.1f} days",
            f"  Total Commission:     {self.total_commission:>14,.2f}",
            "=" * 50,
        ]
        return "\n".join(lines)


class MetricsCalculator:
    """Calculates performance metrics from equity curve and trade records."""

    def calculate(
        self,
        equity_series: pd.Series,
        trades: list[TradeRecord],
        initial_capital: float,
    ) -> MetricsReport:
        """Compute all performance metrics.

        Args:
            equity_series: Time-indexed series of portfolio values.
            trades: List of completed trade records.
            initial_capital: Starting capital.

        Returns:
            MetricsReport with all computed metrics.
        """
        if equity_series.empty:
            return self._empty_report(initial_capital)

        final_value = equity_series.iloc[-1]
        total_return = (final_value / initial_capital) - 1.0

        # Daily returns
        daily_returns = equity_series.pct_change().dropna()

        # Number of trading days
        n_days = len(equity_series)

        # Annual return
        annual_return = self._annual_return(initial_capital, final_value, n_days)

        # Sharpe ratio
        sharpe_ratio = self._sharpe_ratio(daily_returns)

        # Drawdown analysis
        max_drawdown, max_dd_duration = self._drawdown_analysis(equity_series)

        # Trade statistics
        win_rate = self._win_rate(trades)
        profit_loss_ratio = self._profit_loss_ratio(trades)
        avg_holding = self._avg_holding_period(trades)
        total_commission = sum(t.commission for t in trades)

        return MetricsReport(
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            total_trades=len(trades),
            avg_holding_period=avg_holding,
            total_return=total_return,
            total_commission=total_commission,
            initial_capital=initial_capital,
            final_value=final_value,
        )

    # ------------------------------------------------------------------
    # Individual metric computations
    # ------------------------------------------------------------------

    @staticmethod
    def _annual_return(
        initial_capital: float, final_value: float, n_days: int
    ) -> float:
        """Annualized return: (final/initial)^(252/days) - 1."""
        if n_days <= 0 or initial_capital <= 0:
            return 0.0
        return (final_value / initial_capital) ** (
            TRADING_DAYS_PER_YEAR / n_days
        ) - 1.0

    @staticmethod
    def _sharpe_ratio(daily_returns: pd.Series) -> float:
        """Sharpe ratio: mean(r) / std(r) * sqrt(252)."""
        if daily_returns.empty or daily_returns.std() == 0:
            return 0.0
        return (
            daily_returns.mean()
            / daily_returns.std()
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

    @staticmethod
    def _drawdown_analysis(equity_series: pd.Series) -> tuple[float, int]:
        """Calculate max drawdown and its duration in bars.

        Returns:
            (max_drawdown_pct, max_drawdown_duration_bars)
        """
        running_max = equity_series.cummax()
        drawdown = 1.0 - equity_series / running_max

        max_drawdown = drawdown.max()

        # Duration: longest streak where equity is below running max
        is_in_drawdown = drawdown > 0
        if not is_in_drawdown.any():
            return 0.0, 0

        # Find consecutive drawdown periods
        groups = (~is_in_drawdown).cumsum()
        dd_lengths = is_in_drawdown.groupby(groups).sum()
        max_dd_duration = int(dd_lengths.max()) if not dd_lengths.empty else 0

        return float(max_drawdown), max_dd_duration

    @staticmethod
    def _win_rate(trades: list[TradeRecord]) -> float:
        """Win rate: profitable trades / total trades."""
        if not trades:
            return 0.0
        winners = sum(1 for t in trades if t.pnl > 0)
        return winners / len(trades)

    @staticmethod
    def _profit_loss_ratio(trades: list[TradeRecord]) -> float:
        """Profit/loss ratio: avg(winning PnL) / abs(avg(losing PnL))."""
        winners = [t.pnl for t in trades if t.pnl > 0]
        losers = [t.pnl for t in trades if t.pnl < 0]

        if not winners or not losers:
            return 0.0

        avg_win = np.mean(winners)
        avg_loss = abs(np.mean(losers))
        if avg_loss == 0:
            return float("inf")
        return avg_win / avg_loss

    @staticmethod
    def _avg_holding_period(trades: list[TradeRecord]) -> float:
        """Average holding period in days."""
        if not trades:
            return 0.0
        durations = []
        for t in trades:
            if t.entry_time and t.exit_time:
                delta = t.exit_time - t.entry_time
                durations.append(delta.total_seconds() / 86400.0)  # to days
        if not durations:
            return 0.0
        return float(np.mean(durations))

    @staticmethod
    def _empty_report(initial_capital: float) -> MetricsReport:
        return MetricsReport(
            annual_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            win_rate=0.0,
            profit_loss_ratio=0.0,
            total_trades=0,
            avg_holding_period=0.0,
            total_return=0.0,
            total_commission=0.0,
            initial_capital=initial_capital,
            final_value=initial_capital,
        )
