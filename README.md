# Mini Simulation

`mini_simulation` 是一个轻量级研究子项目，用于快速搭建和迭代多智能体金融市场模拟框架。

当前版本已经从“净需求线性冲击价格”的极简市场，升级为一个更接近真实股票市场的单资产批量限价订单簿。它仍然保持代码体量小、便于快速实验，但撮合层已经具备后续继续扩展到 LLM 决策、异质交易者和多资产场景的基础。

## 九宫格实验矩阵

当前项目的核心实验设计仍然是下面这个 `3 x 3` 矩阵：

| 策略类型 \ 决策模式 | `rule_based` | `half_rule_based` | `open_ended` |
| --- | --- | --- | --- |
| `fundamental` | 已接入 | 预留 | 预留 |
| `momentum` | 已接入 | 预留 | 预留 |
| `speculator` | 已接入 | 预留 | 预留 |

也就是说，后续实验始终可以理解为：

- 行：交易者类型
- 列：决策自由度

当前已经稳定跑通的是第一列，也就是三类 agent 的 `rule_based` 基线。

## 当前实现状态

已完成：

- 单资产市场模拟框架
- 三类交易者骨架：
  - `fundamental`
  - `momentum`
  - `speculator`
- `rule_based` 决策模式
- 命令行实验入口
- 正式交易前的 `warm-up` 市场历史注入
- 基于 batch 的限价订单簿
- 价格优先、时间优先的撮合规则
- 未成交订单留存、替换与过期清理
- 实验结果可视化与报告导出
- 使用 `uv` 管理虚拟环境与依赖

尚未完成：

- `half_rule_based`
- `open_ended`
- 更强的 agent 异质性与仓位管理
- 接入真实外部金融数据

## 项目结构

```text
mini_simulation/
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── fundamental.py
│   ├── momentum.py
│   └── speculator.py
├── artifacts/              # 自动生成的实验结果，已加入 gitignore
├── config.py               # 默认实验参数
├── main.py                 # 命令行入口
├── market/
│   ├── __init__.py
│   ├── order_book.py       # 订单簿本体：挂单、排序、撮合
│   └── order_book_market.py # 市场仿真：batch、agent、结算、统计
├── models.py               # 枚举与数据结构
├── prompts.py              # 预留给后续 LLM prompt 逻辑
├── README.md
└── visualization.py        # 图表与实验报告生成
```

## 环境与依赖

推荐环境：

- Python `3.12`
- `uv` 作为环境与依赖管理工具

当前仓库已经包含：

- [pyproject.toml](/D:/Files/undergraduate_research/mini_simulation/pyproject.toml)
- [uv.lock](/D:/Files/undergraduate_research/mini_simulation/uv.lock)
- [.python-version](/D:/Files/undergraduate_research/mini_simulation/.python-version)

当前运行时依赖：

- `matplotlib`

### 使用 uv 创建环境

在项目目录下执行：

```powershell
cd D:\Files\undergraduate_research\mini_simulation
uv sync
```

如果终端里还不能直接识别 `uv`，也可以使用：

```powershell
python -m uv sync
```

### 在 uv 环境中运行命令

推荐直接使用：

```powershell
uv run python main.py
```

也可以在 Windows 下手动激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

## 快速开始

在项目目录下运行：

```powershell
cd D:\Files\undergraduate_research\mini_simulation
uv run python main.py
```

默认会运行：

- `strategy = fundamental`
- `mode = rule_based`
- `steps = 50`
- `agents = 12`

### 命令行参数

示例：

```powershell
uv run python main.py --strategy momentum --mode rule_based --steps 50 --agents 12
uv run python main.py --strategy speculator --mode rule_based --steps 50 --bootstrap-steps 24
```

支持的参数包括：

- `--strategy`：`fundamental | momentum | speculator`
- `--mode`：`rule_based | half_rule_based | open_ended`
- `--steps`：覆盖默认模拟轮数
- `--agents`：覆盖默认智能体数量
- `--bootstrap-steps`：正式交易开始前注入的 warm-up 历史长度

## 输出结果

每次实验运行后会生成：

- 终端摘要
- `summary.json`
- `report.png`

其中 `summary.json` 现在除了基础价格路径之外，还会记录：

- `trade_count_history`
- `batch_count_history`
- `best_bid_history`
- `best_ask_history`
- `open_order_count_history`
- `trade_history`
- 期末仍未成交的买卖挂单

所有结果默认保存到：

```text
mini_simulation/artifacts/
```

该目录已经通过 `.gitignore` 忽略，不进入版本控制。

## 模拟设计

### 1. 每轮先随机切 batch，再按 batch 顺序进入市场

每一轮中，全部 agent 会先被随机打散，再按固定 `batch_size = 3` 切成若干批。

这个设定对应一个很实用的折中：

- 同一个 batch 里的订单视作同一瞬间挂单
- 不同 batch 之间有严格先后
- 默认 `12` 个 agent 时，一轮通常会形成 `4` 个微观撮合片段，既能看到序列效应，也不会把系统复杂度推得过高

### 2. 订单簿遵循价格优先、时间优先

订单簿内部的优先级规则如下：

- 买单：价格高的优先
- 卖单：价格低的优先
- 同价情况下：更早进入簿中的订单优先

时间优先不是用系统真实时钟做的，而是用：

- 第几轮提交
- 第几个 batch 提交
- 进入撮合引擎的顺序号

这三个量联合形成稳定的队列顺序，因此同一轮内也能严格区分先后。

### 3. 同 batch 先取消旧单，再同时提交新单

每个 agent 在自己所属 batch 到来时，会先撤掉自己上一轮残留的旧挂单，再根据当前市场状态重新决策。

这样做的好处是：

- 避免一个 agent 同时挂出过多冲突订单
- 让“本轮的新判断”真正替换旧判断
- 还保留了一个现实细节：在 agent 轮到自己之前，旧挂单仍然有效

### 4. 成交价格由队列中更早的那一侧决定

当最优买价高于或等于最优卖价时，发生撮合。

成交时：

- 成交量取双方剩余量的较小值
- 成交价格采用时间上更早进入订单簿的那一侧报价

这更接近连续限价簿里“被动单先在簿中，主动单去吃单”的逻辑。

### 5. 挂单可以跨轮留存，但不会无限堆积

当前实现里：

- 每个 agent 默认只保留自己最近一次更新后的挂单
- 未成交挂单允许跨轮存在
- 超过 `max_order_age_steps = 3` 的挂单会被清理

这能保留一定的深度与排队效应，但不会让旧订单无限积累、把价格锚死。

### 6. 为了避免同质 agent 场景完全冻住，引入了薄背景流动性

因为当前实验矩阵通常是一整组同类 agent 一起跑，某些时刻会出现“大家同时想买”或“大家同时想卖”的单边市场。

为了让订单簿机制在这种基线实验里也能稳定工作，系统会在每个 batch 注入一层很薄的外部背景流动性。

当前版本采用的是一个很简洁的基线实现：

- 一档被动买单
- 一档被动卖单

它的作用不是替代真实 agent，而是提供一个最小对手盘底座，避免整个市场因为没有反向订单而长期零成交。

### 7. 初始持仓不是零，而是小额等值库存

如果所有 agent 从 `0` 股开始，早期卖出信号几乎无法落地。

因此当前默认设定为：

- 每个 agent 初始持有 `5` 股
- 同时从现金中扣除等值成本
- 初始总财富保持不变

这让买卖两侧都能从第一轮起真实存在。

### 8. 价格更新不再依赖线性净需求冲击

旧版本的价格更新依赖：

- 汇总净需求
- 用线性 `price impact` 直接推下一期价格

现在价格主要由订单簿决定：

- 有成交时，最近成交价会更新市场价格
- 没有成交时，会使用订单簿最优报价构造一个保守的 quote-based mark price

因此 `net_demand_history` 仍然保留，但它的含义已经变成“本轮提交订单的方向性失衡”，而不是直接驱动价格的唯一变量。

### 9. Warm-Up 仍然保留

在正式模拟第 `1` 步开始之前，市场会先注入一段带有不规则涨跌的合成历史。

这段 warm-up 历史会被所有 agent 共享，因此：

- `momentum` 可以读取到趋势与收益率历史
- `fundamental` 即使当前 `rule_based` 仍较简单，也拥有同样的市场背景
- `speculator` 可以继续基于上一轮同伴行为与市场状态做判断

## 可视化

当前生成的图表主要包括：

- 价格路径
- 关键模拟信号叠加
- 净需求与成交量
- 每期买 / 卖 / 观望数量
- 最终 agent 财富分布

图表口径与订单簿版本已经保持一致：

- `volume_history` 表示已成交量
- `net_demand_history` 表示订单方向失衡

## 当前局限

当前版本虽然已经不是极简市场，但仍然是一个刻意控制复杂度的研究基线，主要局限包括：

- 同类 agent 之间仍然比较同质
- `rule_based` 逻辑仍然偏粗
- 外部背景流动性还是简化处理，不是真正建模的做市商
- 暂时只有单资产
- `half_rule_based` 和 `open_ended` 仍是占位实现
- 当前信号仍然是模拟生成的，而不是真实市场数据

## 计划中的下一步

后续更值得推进的方向是：

1. 增加 agent 异质性，包括风险偏好、持仓约束、下单 aggressiveness
2. 完成 `half_rule_based` 和 `open_ended`
3. 重点打磨 LLM 提示词设计与决策接口
4. 将当前模拟信号逐步替换为真实下载的数据
5. 在单资产机制稳定后，再考虑多资产和资金约束联动

## 维护说明

这个子项目的目标仍然是保持模块化、可扩展、便于快速实验。

后续开发时，建议优先遵守这些原则：

- 文件尽量职责单一，不要把所有逻辑堆到一个文件里
- 市场逻辑与 agent 逻辑尽量分离
- 自动生成产物不要提交到版本控制
- 任何影响撮合口径的修改，都要同步更新 `README` 和项目版本号
