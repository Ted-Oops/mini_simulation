"""市场模块。

这个包负责承载订单簿相关的核心实现：

- `OrderBook`：纯撮合层，维护买卖盘、撤单、过期清理和成交。
- `OrderBookMarket`：仿真主循环，负责初始化状态、推进 step 和协调各层组件。
- 其余模块按职责拆分 warm-up、observation、batch 决策、订单生成、背景流动性和 summary。
"""

from market.order_book import OrderBook
from market.order_book_market import OrderBookMarket

__all__ = ["OrderBook", "OrderBookMarket"]
