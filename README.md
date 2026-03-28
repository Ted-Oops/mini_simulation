# Mini Simulation

`mini_simulation` 是一个轻量级研究子项目，用于快速搭建和迭代多智能体金融市场模拟框架。

当前阶段的核心目标，是先搭建一个可扩展、可复现实验、方便后续接入 LLM 的简化市场系统，并逐步完成下面这个 `3 x 3` 实验矩阵：

- `fundamental`
- `momentum`
- `speculator`

分别对应三种决策模式：

- `rule_based`
- `half_rule_based`
- `open_ended`

目前项目重点放在：

- 建立干净的模拟框架
- 跑通 `rule_based` 基线
- 为后续更真实的订单执行机制和 LLM 决策接口做准备

## 九宫格实验矩阵

当前项目的核心实验设计可以直接理解为下面这个九宫格：

| 策略类型 \ 决策模式 | `rule_based` | `half_rule_based` | `open_ended` |
| --- | --- | --- | --- |
| `fundamental` | 已接入 | 预留 | 预留 |
| `momentum` | 已接入 | 预留 | 预留 |
| `speculator` | 已接入 | 预留 | 预留 |

也就是说，后续所有实验都可以理解为：

- 行：交易者类型
- 列：决策自由度

当前已经跑通的是第一列，也就是三类 agent 的 `rule_based` 基线。

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
- 实验结果可视化与报告导出
- 使用 `uv` 管理虚拟环境与依赖

尚未完成：

- `half_rule_based`
- `open_ended`
- 更真实的订单簿 / 撮合机制
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
├── market.py               # 市场循环、warm-up、执行逻辑
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

如果你的终端里还不能直接识别 `uv`，也可以使用：

```powershell
python -m uv sync
```

以上命令会：

- 创建或更新 `.venv`
- 安装运行时依赖
- 按照 `uv.lock` 中锁定的版本同步环境

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

### 命令行参数

示例：

```powershell
uv run python main.py --strategy momentum --mode rule_based --steps 50 --agents 10
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

所有结果默认保存到：

```text
mini_simulation/artifacts/
```

该目录已经通过 `.gitignore` 忽略，不进入版本控制。

## 模拟设计

### 1. 市场机制

当前版本是一个简化的单资产市场。

每一步的流程大致如下：

1. 根据当前市场状态构造 agent observation
2. 每个 agent 生成自己的订单决策
3. 订单按当前市场价格执行
4. 汇总净需求，并通过线性 `price impact` 规则更新下一期价格

这比真实订单簿要简单很多，但足够作为当前阶段的基线系统。

### 2. Warm-Up 历史

在正式模拟第 `1` 步开始之前，市场会先注入一段带有不规则涨跌的合成历史。

这段 warm-up 历史会被所有 agent 共享，因此：

- `momentum` 可以读取到趋势与收益率历史
- `fundamental` 即使在当前 `rule_based` 下不依赖它，也拥有同样的市场背景
- `speculator` 也拥有预热市场上下文

这里有一条重要规则：

- 只有 `speculator` 可以看到“其他人上一期的交易行为”特征
- 其他 agent 可以看到 warm-up 市场历史，但不能读取 peer action 特征

### 3. 三类交易者

#### Fundamental

当前 `rule_based` 版本不再使用潜在价值锚，而是使用模拟生成的 `SUE` 风格信号。

当前逻辑下：

- `SUE` 显著为正时倾向买入
- `SUE` 显著为负时倾向卖出

#### Momentum

当前 `rule_based` 版本主要参考：

- 短期动量
- 中期动量
- 反转分数
- 波动率过滤

#### Speculator

当前 `rule_based` 版本主要参考：

- 上一期其他 agent 的买卖倾向
- 同伴净需求
- 近期价格信息
- `shock` 占位信号

它的行为解释是：

- 投机者通过观察其他人的上一期行为，推断短期市场方向并决定跟随或观望

## 可视化

当前生成的图表主要包括：

- 价格路径
- 关键模拟信号叠加
- 净需求与成交量
- 每期买 / 卖 / 观望数量
- 最终 agent 财富分布

这样做的目标，是让不同 `strategy / mode` 组合的结果更容易横向比较。

## 当前局限

当前版本仍然是一个有意简化的基线系统，主要局限包括：

- 同类 agent 之间仍然比较同质
- `rule_based` 逻辑仍然偏粗
- 订单执行机制还不是完整订单簿
- `half_rule_based` 和 `open_ended` 仍是占位实现
- 当前信号仍然是模拟生成的，而不是真实市场数据

## 计划中的下一步

后续预期路线：

1. 提升执行机制真实性，引入订单簿或更接近真实市场的撮合方式
2. 完成 `half_rule_based` 和 `open_ended`
3. 重点打磨 LLM 提示词设计
4. 将当前模拟信号逐步替换为真实下载的数据

## 维护说明

这个子项目的目标是保持模块化、可扩展、便于快速实验。

后续开发时，建议优先遵守这些原则：

- 文件尽量职责单一，不要把所有逻辑堆到一个文件里
- 市场逻辑与 agent 逻辑尽量分离
- 自动生成产物不要提交到版本控制
