"""MoneyTalks — Streamlit UI for quantitative backtesting and strategy execution."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from moneytalks.backtest.engine import BacktestEngine, BacktestResult
from moneytalks.backtest.metrics import MetricsCalculator, MetricsReport
from moneytalks.data.cleaner import DataCleaner
from moneytalks.data.store import ParquetStore
from moneytalks.data.yfinance_source import YFinanceSource
from moneytalks.data.tushare_source import TushareSource
from moneytalks.storage.database import Database
from moneytalks.strategy.examples.rsi_mean_revert import RSIMeanRevertStrategy
from moneytalks.strategy.examples.sma_cross import SMACrossStrategy

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MoneyTalks",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11 0%, #764ba211 100%);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        color: #666 !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    .stDataFrame { border-radius: 8px; }
    .market-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .market-us { background: #e3f2fd; color: #1565c0; }
    .market-cn { background: #fce4ec; color: #c62828; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Market configuration
# ---------------------------------------------------------------------------
MARKET_CONFIG = {
    "US Stocks (美股)": {
        "key": "us",
        "source_class": YFinanceSource,
        "currency": "$",
        "default_symbol": "AAPL",
        "symbol_help": "e.g. AAPL, MSFT, GOOGL, TSLA",
        "intervals": ["1d", "1h", "30m", "15m", "5m", "1wk"],
        "badge_class": "market-us",
        "badge_label": "US",
    },
    "A-Shares (A股)": {
        "key": "cn",
        "source_class": TushareSource,
        "currency": "¥",
        "default_symbol": "000001.SZ",
        "symbol_help": "e.g. 000001.SZ (平安银行), 600519.SH (贵州茅台)",
        "intervals": ["1d", "1wk", "1mo"],
        "badge_class": "market-cn",
        "badge_label": "A股",
    },
}


# ---------------------------------------------------------------------------
# Singleton resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_yfinance_source():
    return YFinanceSource()


@st.cache_resource
def get_tushare_source(token: str):
    return TushareSource(token=token)


@st.cache_resource
def get_store():
    return ParquetStore()


@st.cache_resource
def get_cleaner():
    return DataCleaner()


@st.cache_resource
def get_db():
    return Database()


def get_data_source(market_key: str, token: str = ""):
    """Return the appropriate DataSource based on market selection."""
    if market_key == "cn":
        return get_tushare_source(token)
    return get_yfinance_source()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.markdown("## MoneyTalks")
    st.sidebar.divider()

    # ── Market Selection ──
    st.sidebar.markdown("**Market / 市场**")
    market_name = st.sidebar.selectbox(
        "Market",
        list(MARKET_CONFIG.keys()),
        index=0,
        label_visibility="collapsed",
    )
    mcfg = MARKET_CONFIG[market_name]

    # Tushare token (only for A-Shares)
    tushare_token = ""
    if mcfg["key"] == "cn":
        tushare_token = st.sidebar.text_input(
            "Tushare Token",
            type="password",
            value=st.session_state.get("tushare_token", ""),
            help="Get your token at https://tushare.pro/user/token",
        )
        st.session_state["tushare_token"] = tushare_token
        if not tushare_token:
            st.sidebar.warning("Please enter your Tushare Pro token to access A-share data.")

    st.sidebar.divider()

    # ── Strategy ──
    strategy_name = st.sidebar.selectbox(
        "Strategy",
        ["SMA Cross (双均线交叉)", "RSI Mean Revert (RSI均值回归)"],
        index=0,
    )

    st.sidebar.divider()

    # ── Symbol & Dates ──
    symbol = st.sidebar.text_input(
        "Symbol",
        value=mcfg["default_symbol"],
        help=mcfg["symbol_help"],
    ).strip().upper()

    col_s, col_e = st.sidebar.columns(2)
    with col_s:
        start_date = st.date_input("Start Date", value=date(2022, 1, 1), min_value=date(2010, 1, 1))
    with col_e:
        end_date = st.date_input("End Date", value=date(2024, 12, 31), max_value=date.today())

    interval = st.sidebar.selectbox("Interval", mcfg["intervals"], index=0)

    st.sidebar.divider()

    # ── Engine Settings ──
    st.sidebar.markdown("**Engine Settings**")
    cur = mcfg["currency"]
    initial_capital = st.sidebar.number_input(
        f"Initial Capital ({cur})", value=100_000, step=10_000, min_value=1_000,
    )
    commission_rate = st.sidebar.number_input(
        "Commission Rate (%)", value=0.10, step=0.01, min_value=0.0, format="%.2f",
    ) / 100.0
    slippage_rate = st.sidebar.number_input(
        "Slippage Rate (%)", value=0.05, step=0.01, min_value=0.0, format="%.2f",
    ) / 100.0

    st.sidebar.divider()

    # ── Strategy Parameters ──
    st.sidebar.markdown("**Strategy Parameters**")
    strategy_params = {}
    if "SMA Cross" in strategy_name:
        strategy_params["fast_period"] = st.sidebar.slider("Fast SMA Period", 3, 50, 10)
        strategy_params["slow_period"] = st.sidebar.slider("Slow SMA Period", 10, 200, 30)
    else:
        strategy_params["rsi_period"] = st.sidebar.slider("RSI Period", 5, 30, 14)
        strategy_params["oversold"] = st.sidebar.slider("Oversold Threshold", 10, 40, 30)
        strategy_params["overbought"] = st.sidebar.slider("Overbought Threshold", 60, 90, 70)

    return {
        "market_name": market_name,
        "market_key": mcfg["key"],
        "market_cfg": mcfg,
        "tushare_token": tushare_token,
        "strategy_name": strategy_name,
        "strategy_params": strategy_params,
        "symbol": symbol,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "interval": interval,
        "initial_capital": float(initial_capital),
        "commission_rate": commission_rate,
        "slippage_rate": slippage_rate,
        "currency": mcfg["currency"],
    }


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_data(
    symbol: str, start: str, end: str, interval: str,
    market_key: str = "us", token: str = "",
) -> pd.DataFrame:
    source = get_data_source(market_key, token)
    store = get_store()
    cleaner = get_cleaner()

    data = source.fetch_historical(symbol, start, end, interval)
    if data.empty:
        return data
    data = cleaner.clean(data, interval)
    store.save(data, symbol, interval)
    if "filled" in data.columns:
        data = data.drop(columns=["filled"])
    return data


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------
def create_strategy(name: str, params: dict, symbol: str):
    params["symbol"] = symbol
    if "SMA Cross" in name:
        return SMACrossStrategy(params)
    return RSIMeanRevertStrategy(params)


# ---------------------------------------------------------------------------
# Currency-aware formatting helpers
# ---------------------------------------------------------------------------
def fmt_money(value: float, currency: str = "$") -> str:
    return f"{currency}{value:,.0f}"


def fmt_money2(value: float, currency: str = "$") -> str:
    return f"{currency}{value:,.2f}"


def fmt_price(value: float, currency: str = "$") -> str:
    return f"{currency}{value:.2f}"


# ---------------------------------------------------------------------------
# Charting helpers (Plotly)
# ---------------------------------------------------------------------------
def plot_candlestick(data: pd.DataFrame, symbol: str, currency: str = "$") -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.75, 0.25],
    )
    fig.add_trace(
        go.Candlestick(
            x=data.index, open=data["open"], high=data["high"],
            low=data["low"], close=data["close"], name="OHLC",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ), row=1, col=1,
    )
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(data["close"], data["open"])]
    fig.add_trace(
        go.Bar(x=data.index, y=data["volume"], name="Volume",
               marker_color=colors, opacity=0.5),
        row=2, col=1,
    )
    fig.update_layout(
        title=f"{symbol} Price", xaxis_rangeslider_visible=False,
        height=500, template="plotly_white",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text=f"Price ({currency})", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def plot_equity_drawdown(equity: pd.Series, initial_capital: float, currency: str = "$") -> go.Figure:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06, row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown"),
    )
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.values, name="Portfolio",
                   line=dict(color="#2196F3", width=1.8),
                   fill="tozeroy", fillcolor="rgba(33,150,243,0.07)"),
        row=1, col=1,
    )
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                  annotation_text="Initial", row=1, col=1)
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown %",
                   line=dict(color="#F44336", width=1.2),
                   fill="tozeroy", fillcolor="rgba(244,67,54,0.15)"),
        row=2, col=1,
    )
    fig.update_layout(height=520, template="plotly_white",
                      margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
    fig.update_yaxes(title_text=f"Value ({currency})", row=1, col=1)
    fig.update_yaxes(title_text="DD %", row=2, col=1)
    return fig


def plot_trade_markers(data: pd.DataFrame, trades, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, y=data["close"], mode="lines",
        name="Close Price", line=dict(color="#9e9e9e", width=1),
    ))
    buy_times = [t.entry_time for t in trades]
    buy_prices = [t.entry_price for t in trades]
    fig.add_trace(go.Scatter(
        x=buy_times, y=buy_prices, mode="markers", name="Buy",
        marker=dict(symbol="triangle-up", size=11, color="#4CAF50",
                    line=dict(width=1, color="white")),
    ))
    sell_times = [t.exit_time for t in trades if t.exit_time]
    sell_prices = [t.exit_price for t in trades if t.exit_price]
    fig.add_trace(go.Scatter(
        x=sell_times, y=sell_prices, mode="markers", name="Sell",
        marker=dict(symbol="triangle-down", size=11, color="#F44336",
                    line=dict(width=1, color="white")),
    ))
    fig.update_layout(
        title=f"{symbol} — Trade Entry / Exit Points",
        height=400, template="plotly_white",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_monthly_heatmap(equity: pd.Series) -> go.Figure | None:
    monthly = equity.resample("ME").last().pct_change().dropna()
    if monthly.empty or len(monthly) < 2:
        return None
    df = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
    pivot = df.pivot_table(values="ret", index="year", columns="month", aggfunc="sum")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[sorted(pivot.columns)]
    text = [[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in pivot.values]
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100, x=month_labels, y=[str(y) for y in pivot.index],
        text=text, texttemplate="%{text}", colorscale="RdYlGn", zmid=0,
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{text}<extra></extra>",
    ))
    fig.update_layout(title="Monthly Returns Heatmap",
                      height=max(200, len(pivot) * 50 + 100),
                      template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ---------------------------------------------------------------------------
# Tab: Backtest
# ---------------------------------------------------------------------------
def tab_backtest(cfg: dict):
    cur = cfg["currency"]
    mcfg = cfg["market_cfg"]
    badge = f'<span class="market-badge {mcfg["badge_class"]}">{mcfg["badge_label"]}</span>'

    run_btn = st.button("🚀  Run Backtest", type="primary", use_container_width=True)

    if run_btn:
        # Validate token for A-shares
        if cfg["market_key"] == "cn" and not cfg["tushare_token"]:
            st.error("Please enter your Tushare Token in the sidebar to access A-share data.")
            return

        with st.spinner(f"Fetching {cfg['symbol']} data..."):
            data = fetch_data(
                cfg["symbol"], cfg["start_date"], cfg["end_date"], cfg["interval"],
                market_key=cfg["market_key"], token=cfg["tushare_token"],
            )
        if data.empty:
            st.error(f"No data returned for **{cfg['symbol']}**. Check the symbol and date range.")
            return

        strategy = create_strategy(cfg["strategy_name"], cfg["strategy_params"], cfg["symbol"])
        engine = BacktestEngine(
            initial_capital=cfg["initial_capital"],
            commission_rate=cfg["commission_rate"],
            slippage_rate=cfg["slippage_rate"],
        )

        with st.spinner("Running backtest..."):
            result = engine.run(strategy, data, symbol=cfg["symbol"], interval=cfg["interval"])

        equity = result.equity_series
        calc = MetricsCalculator()
        metrics = calc.calculate(equity, result.trades, result.initial_capital)

        db = get_db()
        run_id = db.save_backtest(result, metrics)

        st.session_state["last_result"] = result
        st.session_state["last_metrics"] = metrics
        st.session_state["last_data"] = data
        st.session_state["last_run_id"] = run_id
        st.session_state["last_cfg"] = cfg

    # ── Display results ──
    if "last_metrics" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **Run Backtest** to start.")
        return

    metrics = st.session_state["last_metrics"]
    result = st.session_state["last_result"]
    data = st.session_state["last_data"]
    cfg_used = st.session_state["last_cfg"]
    cur = cfg_used.get("currency", "$")
    equity = result.equity_series

    # Header
    pnl = metrics.final_value - metrics.initial_capital
    pnl_color = "green" if pnl >= 0 else "red"
    used_badge_cls = cfg_used["market_cfg"]["badge_class"]
    used_badge_lbl = cfg_used["market_cfg"]["badge_label"]
    st.markdown(
        f'### Results: {result.strategy_name} on {result.symbol} &nbsp; '
        f'<span class="market-badge {used_badge_cls}">{used_badge_lbl}</span> &nbsp; '
        f'<span style="color:{pnl_color}">{metrics.total_return:+.2%}</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Period: {data.index[0].strftime('%Y-%m-%d')} → {data.index[-1].strftime('%Y-%m-%d')} "
        f"| {len(data)} bars | Run #{st.session_state['last_run_id']}"
    )

    # KPI cards
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Final Value", fmt_money(metrics.final_value, cur), delta=f"{metrics.total_return:+.2%}")
    k2.metric("Annual Return", f"{metrics.annual_return:.2%}")
    k3.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.3f}")
    k4.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")
    k5.metric("Win Rate", f"{metrics.win_rate:.1%}", delta=f"{metrics.total_trades} trades")
    k6.metric("Profit/Loss", f"{metrics.profit_loss_ratio:.2f}x")

    st.divider()

    # Charts
    ct1, ct2, ct3, ct4 = st.tabs(
        ["📈 Equity & Drawdown", "🕯️ Price + Trades", "📅 Monthly Returns", "📊 Raw Data"]
    )
    with ct1:
        st.plotly_chart(plot_equity_drawdown(equity, metrics.initial_capital, cur), use_container_width=True)
    with ct2:
        st.plotly_chart(plot_trade_markers(data, result.trades, result.symbol), use_container_width=True)
    with ct3:
        hm = plot_monthly_heatmap(equity)
        if hm:
            st.plotly_chart(hm, use_container_width=True)
        else:
            st.info("Not enough data for monthly heatmap (need at least 2 months).")
    with ct4:
        st.plotly_chart(plot_candlestick(data, result.symbol, cur), use_container_width=True)

    st.divider()

    # Extra metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Holding Period", f"{metrics.avg_holding_period:.1f} days")
    m2.metric("Max DD Duration", f"{metrics.max_drawdown_duration} bars")
    m3.metric("Total Commission", fmt_money2(metrics.total_commission, cur))
    m4.metric("Initial Capital", fmt_money(metrics.initial_capital, cur))

    st.divider()

    # Trade Log
    st.markdown("### Trade Log")
    if result.trades:
        trade_data = []
        for i, t in enumerate(result.trades, 1):
            trade_data.append({
                "#": i,
                "Direction": t.direction,
                "Entry Time": t.entry_time.strftime("%Y-%m-%d") if t.entry_time else "",
                "Entry Price": fmt_price(t.entry_price, cur),
                "Exit Time": t.exit_time.strftime("%Y-%m-%d") if t.exit_time else "",
                "Exit Price": fmt_price(t.exit_price, cur) if t.exit_price else "",
                "Qty": int(t.quantity),
                f"PnL ({cur})": f"{t.pnl:+,.2f}",
                "PnL (%)": f"{t.pnl_pct:+.2%}",
                "Commission": fmt_price(t.commission, cur),
            })
        st.dataframe(
            pd.DataFrame(trade_data), use_container_width=True,
            hide_index=True, height=min(400, len(trade_data) * 38 + 40),
        )
    else:
        st.info("No trades were executed in this backtest.")


# ---------------------------------------------------------------------------
# Tab: History
# ---------------------------------------------------------------------------
def tab_history():
    db = get_db()
    runs = db.list_backtest_runs(limit=50)
    if not runs:
        st.info("No backtest history yet. Run a backtest first.")
        return

    st.markdown("### Backtest History")
    rows = []
    for r in runs:
        rows.append({
            "ID": r.id,
            "Strategy": r.strategy_name,
            "Symbol": r.symbol,
            "Interval": r.interval,
            "Period": f"{r.start_date.strftime('%Y-%m-%d') if r.start_date else '?'} → {r.end_date.strftime('%Y-%m-%d') if r.end_date else '?'}",
            "Capital": f"{r.initial_capital:,.0f}" if r.initial_capital else "",
            "Final": f"{r.final_value:,.0f}" if r.final_value else "",
            "Return": f"{r.annual_return:.2%}" if r.annual_return is not None else "",
            "Sharpe": f"{r.sharpe:.3f}" if r.sharpe is not None else "",
            "MaxDD": f"{r.max_drawdown:.2%}" if r.max_drawdown is not None else "",
            "WinRate": f"{r.win_rate:.1%}" if r.win_rate is not None else "",
            "Trades": r.total_trades if r.total_trades is not None else 0,
            "Date": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 height=min(600, len(rows) * 38 + 40))

    st.divider()
    st.markdown("### Trade Detail Viewer")
    run_ids = [r.id for r in runs]
    selected_id = st.selectbox("Select a backtest run to view trades", run_ids)
    if selected_id:
        trades = db.get_backtest_trades(selected_id)
        if trades:
            trade_rows = []
            for t in trades:
                trade_rows.append({
                    "Symbol": t.symbol,
                    "Direction": t.direction,
                    "Entry": t.entry_time.strftime("%Y-%m-%d") if t.entry_time else "",
                    "Entry Price": f"{t.entry_price:.2f}" if t.entry_price else "",
                    "Exit": t.exit_time.strftime("%Y-%m-%d") if t.exit_time else "",
                    "Exit Price": f"{t.exit_price:.2f}" if t.exit_price else "",
                    "Qty": int(t.quantity) if t.quantity else 0,
                    "PnL": f"{t.pnl:+,.2f}" if t.pnl is not None else "",
                    "PnL%": f"{t.pnl_pct:+.2%}" if t.pnl_pct is not None else "",
                })
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No trades for this run.")


# ---------------------------------------------------------------------------
# Tab: Data Explorer
# ---------------------------------------------------------------------------
def tab_data_explorer(cfg: dict):
    st.markdown("### Data Explorer")
    mcfg = cfg["market_cfg"]
    cur = cfg["currency"]

    col_sym, col_int = st.columns([2, 1])
    with col_sym:
        explore_symbol = st.text_input(
            "Symbol", value=cfg["symbol"], key="explore_sym",
            help=mcfg["symbol_help"],
        ).strip().upper()
    with col_int:
        explore_interval = st.selectbox("Interval", mcfg["intervals"], key="explore_int")

    col_s, col_e, col_btn = st.columns([2, 2, 1])
    with col_s:
        exp_start = st.date_input("Start", value=date(2024, 1, 1), key="exp_start")
    with col_e:
        exp_end = st.date_input("End", value=date.today(), key="exp_end")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("Fetch Data", key="fetch_explore")

    if fetch_btn:
        if cfg["market_key"] == "cn" and not cfg["tushare_token"]:
            st.error("Please enter your Tushare Token in the sidebar.")
            return
        with st.spinner(f"Fetching {explore_symbol}..."):
            data = fetch_data(
                explore_symbol, str(exp_start), str(exp_end), explore_interval,
                market_key=cfg["market_key"], token=cfg["tushare_token"],
            )
        if data.empty:
            st.error("No data returned. Check the symbol and date range.")
            return
        st.session_state["explore_data"] = data
        st.session_state["explore_symbol"] = explore_symbol

    if "explore_data" in st.session_state:
        data = st.session_state["explore_data"]
        sym = st.session_state["explore_symbol"]

        st.plotly_chart(plot_candlestick(data, sym, cur), use_container_width=True)
        st.markdown(
            f"**{len(data)} bars** | "
            f"{data.index[0].strftime('%Y-%m-%d')} → {data.index[-1].strftime('%Y-%m-%d')}"
        )

        with st.expander("View Raw Data"):
            display_df = data.copy()
            display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M")
            st.dataframe(display_df, use_container_width=True, height=400)

        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Latest Close", fmt_price(data["close"].iloc[-1], cur))
        c2.metric("Period High", fmt_price(data["high"].max(), cur))
        c3.metric("Period Low", fmt_price(data["low"].min(), cur))
        pct_change = (data["close"].iloc[-1] / data["close"].iloc[0] - 1)
        c4.metric("Period Return", f"{pct_change:.2%}")
        c5.metric("Avg Volume", f"{data['volume'].mean():,.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = render_sidebar()
    tab1, tab2, tab3 = st.tabs(["⚡ Backtest", "📜 History", "🔍 Data Explorer"])
    with tab1:
        tab_backtest(cfg)
    with tab2:
        tab_history()
    with tab3:
        tab_data_explorer(cfg)


if __name__ == "__main__":
    main()
