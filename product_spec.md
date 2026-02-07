# MoneyTalks - 量化回测与策略执行系统 产品规格说明书

## 1. 项目概述

MoneyTalks 是一个基于 Python 的量化交易框架，支持历史/实时金融数据获取、策略回测、绩效分析和实盘执行。系统以美股市场为初始目标，架构上预留多市场（加密货币、A股、期货等）扩展能力。

### 1.1 核心目标

- **数据获取与管理**：实时和历史金融产品价格获取（分钟级、日级等多种粒度），数据去重、补缺、时间对齐
- **策略回测**：事件驱动回测引擎，支持完整绩效评估（年化收益、夏普比率、最大回撤、胜率、盈亏比、交易次数、平均持仓周期）
- **策略实盘执行**：模拟盘与实盘统一架构，策略代码无需修改即可从回测切换到实盘
- **数据持久化**：行情数据使用 Parquet 高效存储，回测结果和交易记录使用 SQLite 持久化

### 1.2 技术栈

| 类别 | 技术选型 |
|------|----------|
| 语言 | Python 3.11+ |
| 数据获取 | yfinance |
| 数据处理 | pandas, numpy |
| 行情存储 | Parquet (pyarrow) |
| 结果存储 | SQLite (sqlalchemy) |
| 可视化 | matplotlib, seaborn |
| 定时调度 | apscheduler |
| 日志 | loguru |
| 测试 | pytest |

## 2. 系统架构

### 2.1 架构设计原则

采用**事件驱动架构**，核心优势为：回测与实盘共用同一套策略代码，切换时只需更换数据源和执行器，无需修改策略逻辑。

系统分为 5 个核心层：

1. **Data Layer** - 数据获取、清洗、缓存
2. **Strategy Layer** - 策略定义与信号生成
3. **Engine Layer** - 回测引擎与仓位管理
4. **Execution Layer** - 模拟盘/实盘执行
5. **Storage Layer** - 结果持久化

### 2.2 目录结构

```
MoneyTalks/
├── moneytalks/
│   ├── __init__.py
│   ├── config.py                  # 全局配置
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py                # DataSource 抽象基类
│   │   ├── yfinance_source.py     # yfinance 实现
│   │   ├── cleaner.py             # 数据清洗
│   │   └── store.py               # Parquet 存储
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base.py                # Strategy 抽象基类
│   │   ├── signals.py             # Signal/Context 数据类
│   │   └── examples/
│   │       ├── __init__.py
│   │       ├── sma_cross.py       # 双均线交叉策略
│   │       └── rsi_mean_revert.py # RSI 均值回归策略
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py              # 回测引擎
│   │   ├── portfolio.py           # 仓位管理
│   │   ├── metrics.py             # 绩效指标
│   │   └── report.py              # 报告生成
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── base.py                # Broker 抽象基类
│   │   ├── paper.py               # 模拟盘
│   │   └── scheduler.py           # 实盘调度器
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── database.py            # SQLite 连接管理
│   │   └── models.py              # ORM 模型
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # 统一日志
│       └── types.py               # 枚举和类型定义
├── examples/
│   ├── run_backtest.py            # 回测示例
│   └── run_paper_trade.py         # 模拟盘示例
├── tests/
│   ├── test_data.py
│   ├── test_backtest.py
│   └── test_metrics.py
├── data/                          # 本地行情缓存（Parquet）
├── requirements.txt
├── pyproject.toml
└── product_spec.md
```

## 3. 模块详细设计

### 3.1 Data Layer（数据层）

#### 3.1.1 DataSource 抽象基类

定义统一的数据获取接口，便于后续扩展更多数据源（CCXT、AKShare 等）。

**接口方法**：
- `fetch_historical(symbol, start, end, interval)` → `pd.DataFrame`
- `fetch_realtime(symbol)` → `pd.DataFrame`
- `supported_intervals()` → `list[str]`

#### 3.1.2 YFinanceSource

基于 yfinance 库的美股数据获取实现。

**支持粒度**：`1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo`

**数据限制**：分钟级数据仅支持最近 30 天

**输出格式**：标准化 DataFrame，列名统一为 `open, high, low, close, volume`，以时间戳为索引

#### 3.1.3 DataCleaner

数据清洗管道，确保数据质量：

- **去重（deduplicate）**：按时间戳去重，保留最新记录
- **补缺（fill_gaps）**：检测并填补缺失时间点，使用前向填充，同时标记已填充行
- **时间对齐（align_time）**：统一时区为 UTC，将时间戳对齐到标准时间网格

#### 3.1.4 ParquetStore

行情数据本地缓存，使用 Parquet 列式格式高效存储。

- 目录结构：`data/{symbol}/{interval}.parquet`
- 支持增量更新：仅拉取本地缺失的时间段
- 读取时自动合并缓存与新数据

### 3.2 Strategy Layer（策略层）

#### 3.2.1 Strategy 抽象基类

所有策略必须继承此基类并实现 `on_bar` 方法。

**关键方法**：
- `on_init(context)` - 策略初始化，可用于预计算技术指标
- `on_bar(bar, context)` → `Signal | None` - 每根 K 线触发，返回交易信号或 None

#### 3.2.2 Signal 数据类

交易信号的标准化表示：

| 字段 | 类型 | 说明 |
|------|------|------|
| direction | Direction | LONG / SHORT / CLOSE |
| symbol | str | 交易标的 |
| quantity | float | 数量（0 表示由仓位管理器决定） |
| price | float / None | 目标价格（None 表示市价） |
| order_type | OrderType | MARKET / LIMIT |
| reason | str | 信号原因说明 |

#### 3.2.3 StrategyContext

提供策略可安全访问的上下文信息：

- 历史数据窗口（可配置长度）
- 当前持仓信息
- 账户余额
- 当前 bar 索引

#### 3.2.4 示例策略

- **SMA Cross（双均线交叉）**：快线上穿慢线做多，下穿做空
- **RSI Mean Revert（RSI 均值回归）**：RSI 超卖买入，超买卖出

### 3.3 Engine Layer（回测引擎）

#### 3.3.1 BacktestEngine

事件驱动回测核心，逐 bar 遍历历史数据：

1. 初始化策略上下文
2. 遍历每根 K 线，调用策略的 `on_bar` 方法
3. 收到信号后交由 PortfolioManager 执行
4. 回测结束后计算绩效指标并生成报告

#### 3.3.2 PortfolioManager

仓位和资金管理：

- 跟踪现金余额、持仓市值
- 支持做多 / 做空 / 平仓操作
- 可配置手续费率（默认 0.1%）
- 滑点模拟（默认 0.05%）
- 记录完整交易历史

#### 3.3.3 绩效指标（Metrics）

| 指标 | 计算方式 |
|------|----------|
| 年化收益率 | `(最终净值/初始资金)^(252/交易天数) - 1` |
| 夏普比率 | `mean(日收益率) / std(日收益率) * sqrt(252)` |
| 最大回撤 | `max(1 - 净值/历史最高净值)` |
| 胜率 | `盈利交易数 / 总交易数` |
| 盈亏比 | `avg(盈利) / abs(avg(亏损))` |
| 交易次数 | 总开仓次数 |
| 平均持仓周期 | `avg(平仓时间 - 开仓时间)` |

#### 3.3.4 报告生成（Report）

- 终端输出：格式化的文本摘要
- 图表输出：净值曲线、回撤曲线、月度收益热力图（matplotlib）
- 数据持久化：结果自动存入 SQLite

### 3.4 Execution Layer（执行层）

#### 3.4.1 Broker 抽象基类

定义统一的订单执行接口：

- `submit_order(signal)` → `Order`
- `get_position(symbol)` → `Position`
- `get_balance()` → `float`

#### 3.4.2 PaperTrader（模拟盘）

使用实时行情数据模拟下单执行，记录虚拟交易。适用于策略验证阶段。

#### 3.4.3 Scheduler（实盘调度器）

基于 APScheduler 的定时任务调度：

- 按策略指定的时间间隔拉取最新行情
- 调用策略 `on_bar()` 生成信号
- 通过 Broker 接口提交订单
- 所有操作写入日志和数据库

### 3.5 Storage Layer（存储层）

#### 3.5.1 存储方案

- **Parquet**：行情数据（高效列式存储，pandas 原生支持）
- **SQLite**：回测结果、交易记录、运行快照（结构化查询，零配置）

#### 3.5.2 数据模型

**BacktestRun（回测记录）**：
- id, strategy_name, params_json, symbol, interval
- start_date, end_date, initial_capital
- final_value, annual_return, sharpe, max_drawdown
- win_rate, profit_loss_ratio, total_trades, avg_holding_period
- created_at

**Trade（交易记录）**：
- id, backtest_run_id, symbol
- direction, entry_time, entry_price
- exit_time, exit_price, quantity
- pnl, pnl_pct, commission

**LiveSnapshot（实盘快照）**：
- id, strategy_name, timestamp
- portfolio_value, positions_json, pending_orders_json

## 4. 实现计划

Phase 1 按以下顺序逐步实现，每步可独立运行验证：

1. 数据层 - yfinance 数据获取 + 清洗 + Parquet 缓存
2. 策略基类 + 示例策略 - 定义接口，实现 SMA 交叉策略
3. 回测引擎 - 事件驱动循环 + 仓位管理 + 手续费/滑点
4. 绩效指标 + 报告 - 7 大指标计算 + 净值曲线图表
5. SQLite 存储 - 回测结果和交易记录持久化
6. 模拟盘执行 - PaperTrader + 调度器
7. 示例脚本 + 测试 - 端到端可运行的 examples

Phase 2（后续迭代）：
- 自然语言策略描述 → 自动生成策略代码（LLM 集成）

## 5. 扩展性设计

系统通过抽象基类预留了以下扩展点：

- **DataSource**：可扩展 CCXT（加密货币）、AKShare（A股）等数据源
- **Broker**：可对接真实券商 API（如 Alpaca、Interactive Brokers）
- **Strategy**：用户仅需实现 `on_bar` 方法即可创建新策略
- 存储层可按需升级为 PostgreSQL / TimescaleDB
