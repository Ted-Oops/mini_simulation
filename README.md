# 异质交易规则与限价订单簿

本项目实现 Chiarella、Iori、Perelló 的异质交易规则市场模型，用本地仿真核对论文的价格、收益、订单流和订单簿结果。交易者按数学规则决策，无需 API。

**当前结果：500 组核心模拟已完成，部分定性关系得到支持，尚未完整复现全部定量结果。** 正式图表在 [results/baseline](results/baseline)，结论见 [docs/findings.md](docs/findings.md)。

## 项目结构

```text
main.py                  统一运行入口
market/                  项目的市场模型和分析代码
  config.py              参数
  engine.py              预测、需求与市场事件
  order_book.py          价格时间优先撮合、账户与撤单
  statistics.py          收益、尾指数、自相关等统计
  estimators.py          修正R/S和自助区间
  plotting.py            论文对应图表与中文报告
  diagnostics.py         启动阶段及统计口径核查
  fonts.py               中文字体
  storage.py             数据与源代码快照
  cli.py                 命令行及批量运行
results/
  baseline/              已完成的500组正式结果、图像、报告和原始数据
  sensitivity/           参数敏感性试验，包括未匹配和失败分支
  README.md              结果说明与试验索引
docs/
  methodology.md         论文设定、实现假设与核验标准
  findings.md            当前结论
  references/            原论文及核对使用的参考材料
tests/                   市场机制与可重复性测试
pyproject.toml           依赖
uv.lock                  锁定环境
```

## 安装与小样本模拟

在项目目录的 PowerShell 中执行，Python 3.12 或更新版本：

```powershell
uv sync --frozen
uv run --frozen python main.py run --scenarios noise,fundamental,chart12 --seeds 0 --steps 20000 --workers 3 --grid-rule discrete_interval --output results/demo
```

这一条模拟命令会自动生成图像、中文报告和统计核验，无需分别运行三个分析模块。小样本只用于检查程序，不能据此判断论文是否复现。输出目录必须不存在，再次运行请更换目录名。

PyCharm：运行文件选择根目录 `main.py`，工作目录选择本项目，解释器选择 `.venv/Scripts/python.exe`，运行参数填写 `run` 及上述选项。

## 查看已有正式结果

打开 `results/baseline` 中的 PNG 或 Markdown 即可。需要重新生成图表和统计时执行：

```powershell
uv run --frozen python main.py report
# 也可以分析其他已完成批次
uv run --frozen python main.py report results/demo
```

`report` 默认读取 `results/baseline`，不会重新模拟市场。

## 重新运行论文规模的核心比较

```powershell
$seeds = (0..99) -join ','
uv run --frozen python main.py run --seeds $seeds --agents 5000 --steps 200000 --workers 8 --grid-rule discrete_interval --output results/new_baseline
```

默认比较纯噪声、基本面加噪声，以及趋势权重均值 1.2、2.0、2.8，共5种情形；上面命令每种运行100个种子，总计1亿事件。5000人都抽取混合策略权重，并非三类纯策略的人数配比。进程数可按机器资源调为4。

## 参数和输出

```powershell
uv run --frozen python main.py run --help
uv run --frozen python main.py run --scenarios fundamental,chart12 --seeds 100,101,102 --fundamental-horizon 400 --grid-rule discrete_interval --output results/tau400
uv run --frozen python main.py run --scenarios fund05,fundamental,fund15,fund20 --seeds 0,1,2 --grid-rule discrete_interval --output results/fundamental_scan
```

`--steps` 是测量事件数，`--agents` 是总人数，`--seeds` 是逗号分隔的种子，`--burn-in` 显式指定初始剔除期。原文未给 τ_f 数值，`--fundamental-horizon` 默认200是实现假设。`--initialization-steps` 默认0，可单独检查固定基本价值的初始化。其他实现分支及原文歧义见 [方法说明](docs/methodology.md)。`--no-report` 可只模拟，稍后用 `report` 分析。

每个批次保存 `manifest.json`、`source_snapshot.zip`、`results.json`、`runs/<情形>_seed<种子>/`，以及自动生成的中文报告、PNG、CSV和跨种子区间。每个运行目录包含 `config.json`、`events.npz`、`summary.json`，失败时保存 `failure.json`。NPZ 内有价格、基本价值、收益、成交量等逐事件数组。

`success` 表示模拟完成且账户检查通过，不表示论文结果匹配。历史快照保留当时真实代码；当前运行入口始终是根目录 `main.py`。

## 测试

```powershell
uv run --frozen python -m unittest discover -s tests -v
```

覆盖需求方程、成交优先级、部分成交、预留和到期、随机路径重现及独立排序账本的随机撮合对照。
