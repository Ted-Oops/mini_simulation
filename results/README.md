# 实验结果

## 正式核心比较

`baseline/`：5种情形，每种100个种子，5000人、20万事件，共500次运行；500组完成，0组失败。

- [完整结果核对](baseline/复现核对报告.md)
- [启动阶段与尾指数口径核对](baseline/统计口径核对.md)
- [价格与收益图](baseline/图2_3_价格与收益.png)
- [波动持续性图](baseline/图5_波动相关与修正极差.png)
- [当前结论](../docs/findings.md)

这些数据支持部分定性关系，尚未完整匹配原文。程序完成状态不等于科学结果匹配。

## 敏感性试验

以下均保留配置、种子、源代码快照和运行结果，包括失败或未匹配的分支。它们用于核查论文缺项和实现选择，不代替正式核心比较。

- `sensitivity/grid_round_fullpilot/`
- `sensitivity/history_trait_crosscheck/`
- `sensitivity/individual_low_noise_diagnostic/`
- `sensitivity/individual_tau_fullpilot/`
- `sensitivity/initialization_sweep/`
- `sensitivity/integer_fixed_fullpilot/`
- `sensitivity/integer_individual_fullpilot/`
- `sensitivity/long_initialization_diagnostic/`
- `sensitivity/market_warmup_fixed_diagnostic/`
- `sensitivity/market_warmup_individual_diagnostic/`
- `sensitivity/normalized_population_clock_diagnostic/`
- `sensitivity/normalized_traits_diagnostic/`
- `sensitivity/original_std_fullpilot/`
- `sensitivity/pilot_std/`
- `sensitivity/pilot_variance/`
- `sensitivity/replace_fullpilot/`
- `sensitivity/tau_f_sweep/`
- `sensitivity/unreported_tau_calibration/`

`baseline/source_snapshot.zip` 记录该批次运行时的真实代码，保留历史模块名；当前源代码结构为根目录 `market/`。数值路径未因目录整理改变。重新分析执行 `python main.py report`，新模拟执行 `python main.py run`。
