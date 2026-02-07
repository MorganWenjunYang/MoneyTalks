"""Report generation: text summaries and matplotlib charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from moneytalks.backtest.engine import BacktestResult
from moneytalks.backtest.metrics import MetricsCalculator, MetricsReport
from moneytalks.utils.logger import get_logger

logger = get_logger("backtest.report")


class ReportGenerator:
    """Generates text reports and chart visualizations from backtest results."""

    def __init__(self):
        self.metrics_calculator = MetricsCalculator()

    def generate(
        self,
        result: BacktestResult,
        save_charts: bool = True,
        chart_dir: str | Path | None = None,
    ) -> MetricsReport:
        """Generate a full report from backtest results.

        Args:
            result: BacktestResult from the engine.
            save_charts: Whether to save chart images.
            chart_dir: Directory to save charts. Defaults to current dir.

        Returns:
            MetricsReport with all computed metrics.
        """
        equity = result.equity_series
        trades = result.trades

        # Calculate metrics
        metrics = self.metrics_calculator.calculate(
            equity_series=equity,
            trades=trades,
            initial_capital=result.initial_capital,
        )

        # Print text summary
        print(metrics.summary())
        print(f"  Strategy: {result.strategy_name}")
        print(f"  Symbol:   {result.symbol}")
        print(f"  Interval: {result.interval}")
        print(f"  Period:   {result.data.index[0]} -> {result.data.index[-1]}")
        print("=" * 50)

        # Generate charts
        if save_charts and not equity.empty:
            chart_path = Path(chart_dir) if chart_dir else Path(".")
            chart_path.mkdir(parents=True, exist_ok=True)
            self._plot_equity_and_drawdown(equity, metrics, result, chart_path)
            self._plot_monthly_returns(equity, result, chart_path)
            logger.info(f"Charts saved to {chart_path}")

        return metrics

    # ------------------------------------------------------------------
    # Chart generation
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_equity_and_drawdown(
        equity: pd.Series,
        metrics: MetricsReport,
        result: BacktestResult,
        chart_path: Path,
    ):
        """Plot equity curve and drawdown on a 2-panel figure."""
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )

        # Equity curve
        ax1.plot(equity.index, equity.values, linewidth=1.2, color="#2196F3")
        ax1.axhline(
            y=metrics.initial_capital, color="gray", linestyle="--", alpha=0.5
        )
        ax1.fill_between(
            equity.index,
            metrics.initial_capital,
            equity.values,
            where=equity.values >= metrics.initial_capital,
            alpha=0.15,
            color="#4CAF50",
        )
        ax1.fill_between(
            equity.index,
            metrics.initial_capital,
            equity.values,
            where=equity.values < metrics.initial_capital,
            alpha=0.15,
            color="#F44336",
        )
        ax1.set_title(
            f"{result.strategy_name} | {result.symbol} | "
            f"Return: {metrics.total_return:.2%} | "
            f"Sharpe: {metrics.sharpe_ratio:.2f}",
            fontsize=13,
        )
        ax1.set_ylabel("Portfolio Value")
        ax1.grid(True, alpha=0.3)
        ax1.legend(["Equity", "Initial Capital"], loc="upper left")

        # Drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        ax2.fill_between(
            drawdown.index, drawdown.values, 0, alpha=0.4, color="#F44336"
        )
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

        plt.tight_layout()
        filepath = chart_path / f"{result.strategy_name}_{result.symbol}_equity.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _plot_monthly_returns(
        equity: pd.Series,
        result: BacktestResult,
        chart_path: Path,
    ):
        """Plot a heatmap of monthly returns."""
        # Calculate monthly returns
        monthly = equity.resample("ME").last().pct_change().dropna()

        if monthly.empty or len(monthly) < 2:
            return

        # Build year x month matrix
        returns_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })

        pivot = returns_df.pivot_table(
            values="return", index="year", columns="month", aggfunc="sum"
        )

        # Rename columns to month abbreviations
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]

        fig, ax = plt.subplots(figsize=(12, max(3, len(pivot) * 0.6 + 1)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1%",
            center=0,
            cmap="RdYlGn",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(
            f"{result.strategy_name} | {result.symbol} | Monthly Returns",
            fontsize=13,
        )
        ax.set_ylabel("Year")
        ax.set_xlabel("")

        plt.tight_layout()
        filepath = chart_path / f"{result.strategy_name}_{result.symbol}_monthly.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
