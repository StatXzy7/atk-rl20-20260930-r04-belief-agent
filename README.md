# atk-rl20-20260930-r04-belief-agent

有限预算信息获取 Agent：以 `(后验信念, 剩余预算)` 为状态的精确
Bellman 规划器，含基线、独立枚举核验、采样对照与失败反例。
全部概率用 `fractions.Fraction` 精确表示，不依赖任何外部
语言模型或工具服务。

## 运行

```
python3 demo.py                 # 关键数值一览（精确有理数）
python3 -m pytest tests/ -q     # 35 个测试
```

## 结构

- `budgeted_info_agent/model.py` — 问题定义（先验、工具、κ、预算）与校验
- `budgeted_info_agent/belief.py` — 贝叶斯更新、预测概率、停止价值
- `budgeted_info_agent/planner.py` — 精确规划器（Fraction 缓存键）与
  浮点舍入键对照实现
- `budgeted_info_agent/baselines.py` — 立即停止 / 固定次序 / 单步贪心，
  及任意无记忆策略的精确期望收益
- `budgeted_info_agent/enumerate.py` — 无缓存树展开、全策略暴力枚举
- `budgeted_info_agent/simulate.py` — 固定种子采样与可重算回合记录
- `budgeted_info_agent/examples.py` — parity 差距例子、信息增益陷阱、
  工具相关性反例
- `DERIVATION.md` — 推导、缓存风险、状态空间增长、相关性分析
