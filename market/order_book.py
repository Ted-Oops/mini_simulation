"""订单簿实现。

这个文件只关心“订单如何进入簿中、如何排序、如何撮合、如何移除”，
不关心 agent 如何思考，也不关心每轮实验如何推进。
"""

from models import LimitOrder, TradeAction, TradeExecution


class OrderBook:
    """单资产限价订单簿。

    这个类是一个纯撮合组件，负责维护：

    - 买单队列
    - 卖单队列
    - agent 当前仍在簿中的挂单索引
    - 已发生的成交记录

    它遵循两个最核心的市场规则：

    - 价格优先：更优价格先成交
    - 时间优先：同价订单按更早进入系统的先后成交
    """

    def __init__(self, initial_price: float, price_tick: float) -> None:
        """初始化订单簿状态。

        参数：
        - `initial_price`：市场初始参考价，用于无成交时的价格锚。
        - `price_tick`：最小价格变动单位，用于报价离散化。
        """

        self.last_trade_price = initial_price
        self.price_tick = max(price_tick, 0.01)
        self.buy_orders: list[LimitOrder] = []
        self.sell_orders: list[LimitOrder] = []
        self.agent_orders: dict[str, LimitOrder] = {}
        self.trades: list[TradeExecution] = []

    @staticmethod
    def _buy_priority(order: LimitOrder) -> tuple[float, int, int, int]:
        """返回买单的排序键。

        逻辑是：
        - 买价越高越优先
        - 若价格相同，则更早提交的订单优先
        """

        return (
            -order.limit_price,
            order.submitted_step,
            order.submitted_batch,
            order.sequence,
        )

    @staticmethod
    def _sell_priority(order: LimitOrder) -> tuple[float, int, int, int]:
        """返回卖单的排序键。

        逻辑是：
        - 卖价越低越优先
        - 若价格相同，则更早提交的订单优先
        """

        return (
            order.limit_price,
            order.submitted_step,
            order.submitted_batch,
            order.sequence,
        )

    @staticmethod
    def _time_priority(order: LimitOrder) -> tuple[int, int, int]:
        """提取订单的时间优先键。

        这里不使用真实时钟，而是使用：
        - 第几轮
        - 第几个 batch
        - 全局顺序号 sequence

        这样可以保证仿真中时间优先规则稳定且可复现。
        """

        return order.submitted_step, order.submitted_batch, order.sequence

    def _sort_books(self) -> None:
        """对买卖盘重新排序。"""

        self.buy_orders.sort(key=self._buy_priority)
        self.sell_orders.sort(key=self._sell_priority)

    def add_order(self, order: LimitOrder) -> None:
        """把一笔新订单加入订单簿。

        加入后会同步更新：
        - agent 到挂单的索引
        - 买卖盘队列顺序
        """

        self.agent_orders[order.agent_id] = order
        if order.action == TradeAction.BUY:
            self.buy_orders.append(order)
        else:
            self.sell_orders.append(order)
        self._sort_books()

    def cancel_agent_order(self, agent_id: str) -> LimitOrder | None:
        """撤销指定 agent 当前仍在簿中的挂单。

        一个 agent 在当前实现里默认只保留一笔活动挂单。
        如果该 agent 没有未成交订单，则返回 `None`。
        """

        existing_order = self.agent_orders.pop(agent_id, None)
        if existing_order is None:
            return None

        if existing_order.action == TradeAction.BUY:
            self.buy_orders = [
                order
                for order in self.buy_orders
                if order.order_id != existing_order.order_id
            ]
        else:
            self.sell_orders = [
                order
                for order in self.sell_orders
                if order.order_id != existing_order.order_id
            ]
        return existing_order

    def clear_expired_orders(self, current_step: int, max_age_steps: int) -> int:
        """清理超过最大存活期的旧挂单。

        参数：
        - `current_step`：当前要进入的正式轮次
        - `max_age_steps`：挂单最多允许跨多少轮保留

        返回值：
        - 本次被清理掉的订单数量
        """

        if max_age_steps <= 0:
            expired_orders = len(self.buy_orders) + len(self.sell_orders)
            self.buy_orders = []
            self.sell_orders = []
            self.agent_orders.clear()
            return expired_orders

        expired_count = 0

        remaining_buy_orders: list[LimitOrder] = []
        for order in self.buy_orders:
            if current_step - order.submitted_step >= max_age_steps:
                self.agent_orders.pop(order.agent_id, None)
                expired_count += 1
            else:
                remaining_buy_orders.append(order)

        remaining_sell_orders: list[LimitOrder] = []
        for order in self.sell_orders:
            if current_step - order.submitted_step >= max_age_steps:
                self.agent_orders.pop(order.agent_id, None)
                expired_count += 1
            else:
                remaining_sell_orders.append(order)

        self.buy_orders = remaining_buy_orders
        self.sell_orders = remaining_sell_orders
        self._sort_books()
        return expired_count

    def best_bid(self) -> float | None:
        """返回当前最优买价。"""

        if not self.buy_orders:
            return None
        return self.buy_orders[0].limit_price

    def best_ask(self) -> float | None:
        """返回当前最优卖价。"""

        if not self.sell_orders:
            return None
        return self.sell_orders[0].limit_price

    def order_count(self) -> int:
        """返回当前订单簿中的活动挂单总数。"""

        return len(self.buy_orders) + len(self.sell_orders)

    def reference_price(self) -> float:
        """返回当前市场参考价。

        逻辑分三种情况：
        - 两边盘口都存在：取最优买卖价中点
        - 只有单边盘口：用最近成交价和该单边报价做保守加权
        - 完全无挂单：退回最近成交价
        """

        best_bid = self.best_bid()
        best_ask = self.best_ask()

        if best_bid is not None and best_ask is not None:
            return self._align_mark_price((best_bid + best_ask) / 2)
        if best_bid is not None:
            return self._align_mark_price((2 * self.last_trade_price + best_bid) / 3)
        if best_ask is not None:
            return self._align_mark_price((2 * self.last_trade_price + best_ask) / 3)
        return self.last_trade_price

    def _align_mark_price(self, raw_price: float) -> float:
        """把价格吸附到 tick 网格上。"""

        tick_count = round(raw_price / self.price_tick)
        return max(tick_count * self.price_tick, self.price_tick)

    def _execution_price(
        self, buy_order: LimitOrder, sell_order: LimitOrder
    ) -> float:
        """决定一笔撮合的成交价格。

        当前采用的规则是：
        谁更早进入簿中，就沿用谁的报价作为成交价。

        这相当于把更早在簿中的订单视为被动单，
        而后来的订单视为主动吃单的一方。
        """

        if self._time_priority(buy_order) <= self._time_priority(sell_order):
            return buy_order.limit_price
        return sell_order.limit_price

    def match_orders(self, step: int, batch: int) -> list[TradeExecution]:
        """执行一次连续撮合。

        只要最优买价大于等于最优卖价，就持续成交，直到盘口不再交叉。
        每次成交都会：
        - 生成一条 `TradeExecution`
        - 减少双方剩余数量
        - 在订单数量归零时将其移出订单簿
        """

        self._sort_books()
        new_trades: list[TradeExecution] = []
        trade_sequence = 0

        while self.buy_orders and self.sell_orders:
            buy_order = self.buy_orders[0]
            sell_order = self.sell_orders[0]

            if buy_order.limit_price < sell_order.limit_price:
                break

            trade_quantity = min(buy_order.quantity, sell_order.quantity)
            execution_price = self._execution_price(buy_order, sell_order)

            trade = TradeExecution(
                trade_id=f"trade_{step}_{batch}_{trade_sequence}",
                buy_order_id=buy_order.order_id,
                sell_order_id=sell_order.order_id,
                price=execution_price,
                quantity=trade_quantity,
                step=step,
                batch=batch,
                sequence=trade_sequence,
                buyer_id=buy_order.agent_id,
                seller_id=sell_order.agent_id,
            )
            self.trades.append(trade)
            new_trades.append(trade)
            self.last_trade_price = execution_price

            buy_order.quantity -= trade_quantity
            sell_order.quantity -= trade_quantity
            trade_sequence += 1

            if buy_order.quantity == 0:
                self.buy_orders.pop(0)
                self.agent_orders.pop(buy_order.agent_id, None)

            if sell_order.quantity == 0:
                self.sell_orders.pop(0)
                self.agent_orders.pop(sell_order.agent_id, None)

        return new_trades
