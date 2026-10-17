# 异质交易规则限价订单簿

基于 Chiarella、Iori、Perelló 的模型进行本地市场模拟，不使用大模型 API。源码位于 `market/`，入口为 `main.py`。

## 运行

Python 3.12 或更新版本，在项目目录执行：

```powershell
uv sync --frozen
uv run --frozen python main.py run --scenarios noise,fundamental,chart12 --seeds 0 --steps 20000 --workers 3 --grid-rule discrete_interval --output results/demo
```

论文规模的核心比较：

```powershell
$seeds = (0..99) -join ','
uv run --frozen python main.py run --seeds $seeds --agents 5000 --steps 200000 --workers 8 --grid-rule discrete_interval --output results/new_baseline
```

输出目录必须不存在。参数列表：`uv run --frozen python main.py run --help`。

## 结果

`results/baseline/` 保存已有500组运行的最终结果：

- `figures/`：7张图像。
- `config.json`：实验配置、种子与运行状态。
- `statistics.csv`：逐种子统计。
- `statistics.json`：分组估计、配对差异与分阶段统计。

这些图表来自5种情形、每种100个种子、5000人和20万事件的模拟。部分定性关系得到支持，但价格偏离等结果仍未与原文完全对齐。原文未给出 τ_f 的数值，当前假设为200；噪声标准差采用1e-4，其他设置见 `config.json`。

模拟结束后默认清理中间数据，只保留最终图像、配置和统计。需要保留逐事件数据时添加 `--keep-data`，再使用 `python main.py report results/批次名` 重新分析。已提供的baseline原始数据已清理，重新模拟可生成新的结果。

## 测试

```powershell
uv run --frozen python -m unittest discover -s tests -v
```
