"""市场模块。

这个包负责承载订单簿相关的核心实现：

- `OrderBook`：纯撮合层，维护买卖盘、撤单、过期清理和成交。
- `OrderBookMarket`：仿真层，负责 agent、batch、观测构造和整轮市场推进。
"""

from market.order_book import OrderBook
from market.order_book_market import OrderBookMarket

__all__ = ["OrderBook", "OrderBookMarket"]
