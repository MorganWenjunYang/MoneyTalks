# MoneyTalks

量化回测与策略执行系统 — 基于 Python 的事件驱动量化交易框架，支持历史数据回测、绩效分析和模拟盘执行。

## 功能特性

- 美股 / A股数据获取（yfinance / tushare），支持多种时间粒度
- 数据自动清洗（去重、补缺、时间对齐）+ Parquet 本地缓存
- 事件驱动回测引擎，含手续费 & 滑点模拟
- 完整绩效指标：年化收益、夏普比率、最大回撤、胜率、盈亏比等
- 模拟盘实时执行（策略代码回测 / 实盘零修改切换）
- Streamlit Web UI — K 线图、净值曲线、回撤图、月度收益热力图
- 回测结果 & 交易记录 SQLite 持久化

## 安装

```bash
# 克隆项目
git clone <repo-url> && cd MoneyTalks

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 运行 Web UI

```bash
streamlit run app.py
```

### 命令行回测

```bash
python examples/run_backtest.py
```

### 命令行模拟盘

```bash
python examples/run_paper_trade.py
```

## 如何添加自定义策略

MoneyTalks 采用面向对象的策略架构。你只需要 **继承 `Strategy` 基类，实现 `on_bar` 方法**，即可接入回测引擎和模拟盘，无需修改框架代码。

### 第一步：了解核心概念

| 概念 | 说明 |
|------|------|
| `Strategy` | 策略抽象基类，所有策略必须继承它 |
| `on_bar(bar, context)` | 每根 K 线触发一次，返回交易信号或 `None` |
| `on_init(context)` | 可选，回测开始前调用一次，用于预计算指标 |
| `Signal` | 交易信号，包含方向、标的、数量、价格等信息 |
| `StrategyContext` | 策略上下文，提供历史数据、持仓信息、账户余额 |
| `Direction` | 交易方向枚举：`LONG`（做多）、`SHORT`（做空）、`CLOSE`（平仓） |

### 第二步：创建策略文件

在 `moneytalks/strategy/examples/` 目录下新建一个 Python 文件，例如 `my_strategy.py`：

```python
"""我的自定义策略示例 — 布林带突破策略。

当价格突破布林带上轨时做多，跌破下轨时平仓。
"""

from __future__ import annotations

import pandas as pd

from moneytalks.strategy.base import Strategy
from moneytalks.strategy.signals import Signal, StrategyContext
from moneytalks.utils.types import Direction, OrderType


class BollingerBreakoutStrategy(Strategy):
    """布林带突破策略。

    Parameters (通过 params 字典传入):
        period (int): 布林带周期，默认 20。
        num_std (float): 标准差倍数，默认 2.0。
        symbol (str): 交易标的，默认 "AAPL"。
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.period: int = self.params.get("period", 20)
        self.num_std: float = self.params.get("num_std", 2.0)
        self.symbol: str = self.params.get("symbol", "AAPL")

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        history = context.history

        # 数据不足时跳过
        if len(history) < self.period:
            return None

        close = history["close"]
        sma = close.rolling(self.period).mean().iloc[-1]
        std = close.rolling(self.period).std().iloc[-1]
        upper_band = sma + self.num_std * std
        lower_band = sma - self.num_std * std

        current_price = bar["close"]
        position = context.get_position(self.symbol)

        # 价格突破上轨 → 做多
        if current_price > upper_band and (position is None or not position.is_long):
            return Signal(
                direction=Direction.LONG,
                symbol=self.symbol,
                order_type=OrderType.MARKET,
                reason=f"突破上轨: price={current_price:.2f} > upper={upper_band:.2f}",
                timestamp=bar.name if hasattr(bar, "name") else None,
            )

        # 价格跌破下轨 → 平仓
        if current_price < lower_band and position is not None and position.is_long:
            return Signal(
                direction=Direction.CLOSE,
                symbol=self.symbol,
                order_type=OrderType.MARKET,
                reason=f"跌破下轨: price={current_price:.2f} < lower={lower_band:.2f}",
                timestamp=bar.name if hasattr(bar, "name") else None,
            )

        return None
```

### 第三步：理解 `on_bar` 的参数和返回值

**`bar`** — 当前这根 K 线的数据（`pd.Series`），包含以下字段：

| 字段 | 说明 |
|------|------|
| `bar["open"]` | 开盘价 |
| `bar["high"]` | 最高价 |
| `bar["low"]` | 最低价 |
| `bar["close"]` | 收盘价 |
| `bar["volume"]` | 成交量 |
| `bar.name` | 时间戳（`datetime`） |

**`context`** — 策略上下文（`StrategyContext`），常用属性和方法：

| 属性 / 方法 | 说明 |
|-------------|------|
| `context.history` | 截至当前 bar 的所有历史数据（`pd.DataFrame`） |
| `context.current_bar` | 当前 bar（同 `bar` 参数） |
| `context.cash` | 当前可用现金 |
| `context.portfolio_value` | 总资产（现金 + 持仓市值） |
| `context.get_position(symbol)` | 获取某标的的持仓，返回 `Position` 或 `None` |
| `context.positions` | 所有持仓的字典 `{symbol: Position}` |
| `context.initial_capital` | 初始资金 |

**返回值** — `Signal` 或 `None`：

- 返回 `None` 表示本根 K 线不操作
- 返回 `Signal` 表示产生交易信号，引擎会自动执行下单

**`Signal` 的字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `direction` | `Direction` | `LONG`（做多）/ `SHORT`（做空）/ `CLOSE`（平仓） |
| `symbol` | `str` | 交易标的 |
| `quantity` | `float` | 交易数量，`0` 表示由仓位管理器自动决定 |
| `price` | `float \| None` | 限价，`None` 表示市价单 |
| `order_type` | `OrderType` | `MARKET`（市价）/ `LIMIT`（限价） |
| `reason` | `str` | 信号原因说明（会记录在交易日志中） |
| `timestamp` | `datetime \| None` | 信号时间戳 |

### 第四步：在回测脚本中使用自定义策略

```python
from moneytalks.backtest.engine import BacktestEngine
from moneytalks.data.yfinance_source import YFinanceSource
from moneytalks.data.cleaner import DataCleaner
from moneytalks.strategy.examples.my_strategy import BollingerBreakoutStrategy

# 获取数据
source = YFinanceSource()
cleaner = DataCleaner()
data = source.fetch_historical("AAPL", "2022-01-01", "2024-12-31", "1d")
data = cleaner.clean(data, "1d")
if "filled" in data.columns:
    data = data.drop(columns=["filled"])

# 创建策略（通过 params 字典传入参数）
strategy = BollingerBreakoutStrategy(params={
    "period": 20,
    "num_std": 2.0,
    "symbol": "AAPL",
})

# 运行回测
engine = BacktestEngine(initial_capital=100_000.0)
result = engine.run(strategy=strategy, data=data, symbol="AAPL", interval="1d")

# 查看结果
print(f"Final value: ${result.equity_series.iloc[-1]:,.2f}")
print(f"Total trades: {len(result.trades)}")
```

### 第五步（可选）：利用 `on_init` 预计算指标

如果你的策略需要在回测开始前做一些预计算（例如提前算好所有技术指标以提高性能），可以重写 `on_init` 方法：

```python
class MyOptimizedStrategy(Strategy):

    def on_init(self, context: StrategyContext) -> None:
        """回测开始前调用，预计算所有指标。"""
        close = context.data["close"]
        self._sma = close.rolling(20).mean()
        self._upper = self._sma + 2 * close.rolling(20).std()
        self._lower = self._sma - 2 * close.rolling(20).std()

    def on_bar(self, bar: pd.Series, context: StrategyContext) -> Signal | None:
        idx = context.current_bar_index
        if pd.isna(self._sma.iloc[idx]):
            return None

        price = bar["close"]
        upper = self._upper.iloc[idx]
        lower = self._lower.iloc[idx]

        # ... 策略逻辑 ...
        return None
```

### 策略开发最佳实践

1. **参数化**：所有可调参数通过 `params` 字典传入，便于后续参数优化
2. **数据安全检查**：在 `on_bar` 开头检查历史数据长度是否足够
3. **填写 `reason`**：在 Signal 中写明触发原因，方便回测后分析交易日志
4. **不要直接修改 context**：`StrategyContext` 是只读的，策略只应返回 Signal
5. **幂等性**：同一根 K 线数据多次调用 `on_bar` 应返回相同结果

### 内置策略参考

框架自带两个示例策略，可作为开发参考：

| 策略 | 文件 | 逻辑 |
|------|------|------|
| SMA Cross（双均线交叉） | `moneytalks/strategy/examples/sma_cross.py` | 快线上穿慢线做多，下穿平仓 |
| RSI Mean Revert（RSI 均值回归） | `moneytalks/strategy/examples/rsi_mean_revert.py` | RSI 超卖买入，超买卖出 |

## 项目结构

```
MoneyTalks/
├── moneytalks/
│   ├── config.py                  # 全局配置
│   ├── data/
│   │   ├── base.py                # DataSource 抽象基类
│   │   ├── yfinance_source.py     # yfinance 数据源（美股）
│   │   ├── tushare_source.py      # tushare 数据源（A股）
│   │   ├── cleaner.py             # 数据清洗
│   │   └── store.py               # Parquet 本地缓存
│   ├── strategy/
│   │   ├── base.py                # Strategy 抽象基类
│   │   ├── signals.py             # Signal / StrategyContext 数据类
│   │   └── examples/
│   │       ├── sma_cross.py       # 双均线交叉策略
│   │       └── rsi_mean_revert.py # RSI 均值回归策略
│   ├── backtest/
│   │   ├── engine.py              # 回测引擎
│   │   ├── portfolio.py           # 仓位管理
│   │   ├── metrics.py             # 绩效指标计算
│   │   └── report.py              # 报告生成
│   ├── execution/
│   │   ├── base.py                # Broker 抽象基类
│   │   ├── paper.py               # 模拟盘执行
│   │   └── scheduler.py           # 实盘调度器
│   ├── storage/
│   │   ├── database.py            # SQLite 数据库
│   │   └── models.py              # ORM 模型
│   └── utils/
│       ├── logger.py              # 日志
│       └── types.py               # 枚举和类型定义
├── examples/
│   ├── run_backtest.py            # 回测示例脚本
│   └── run_paper_trade.py         # 模拟盘示例脚本
├── tests/                         # 测试
├── app.py                         # Streamlit Web UI
├── requirements.txt
└── product_spec.md
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| 数据获取 | yfinance, tushare |
| 数据处理 | pandas, numpy |
| 行情存储 | Parquet (pyarrow) |
| 结果存储 | SQLite (sqlalchemy) |
| 可视化 | plotly, matplotlib, seaborn |
| Web UI | streamlit |
| 定时调度 | apscheduler |
| 日志 | loguru |
| 测试 | pytest |

## License

MIT
