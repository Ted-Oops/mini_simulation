import math
import random
from collections.abc import Callable

from market.order_book import OrderBook
from market.orders import align_limit_price
from models import ExperimentConfig, LimitOrder, TradeAction


class BackgroundLiquidityManager:
    """维护跨 batch 留存的背景买卖单。"""

    REFRESH_INTERVAL_BATCHES = 2
    REPRICE_THRESHOLD_TICKS = 3

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        rng: random.Random,
        order_book: OrderBook,
        next_sequence: Callable[[], int],
        anchor_price: float,
    ) -> None:
        self.config = config
        self.rng = rng
        self.order_book = order_book
        self.next_sequence = next_sequence
        self.bid_agent_id = "background_bid"
        self.ask_agent_id = "background_ask"
        self.anchor_price = anchor_price
        self.batch_clock = 0
        self.last_refresh_clock = -self.REFRESH_INTERVAL_BATCHES

    def maintain(
        self,
        *,
        step: int,
        batch: int,
        reference_price: float,
    ) -> None:
        """在 batch 开始前补充或刷新背景流动性。"""

        self.batch_clock += 1
        anchor_price = max(reference_price, self.config.price_tick)
        current_bid_order, current_ask_order = self._orders()
        if not self._should_refresh(
            anchor_price,
            current_bid_order,
            current_ask_order,
        ):
            return

        self._replace(step, batch, anchor_price)

    def realign_after_trades(
        self,
        *,
        step: int,
        batch: int,
        reference_price: float,
    ) -> None:
        """当最新成交价打穿旧背景报价时，立即按新价位重挂。"""

        anchor_price = max(reference_price, self.config.price_tick)
        current_bid_order, current_ask_order = self._orders()
        if (
            current_bid_order is None
            or current_ask_order is None
            or current_bid_order.limit_price >= anchor_price
            or current_ask_order.limit_price <= anchor_price
        ):
            self._replace(step, batch, anchor_price)

    def _orders(self) -> tuple[LimitOrder | None, LimitOrder | None]:
        """读取当前仍留在簿中的背景买卖单。"""

        bid_order = self.order_book.agent_orders.get(self.bid_agent_id)
        ask_order = self.order_book.agent_orders.get(self.ask_agent_id)
        return bid_order, ask_order

    def _should_refresh(
        self,
        anchor_price: float,
        bid_order: LimitOrder | None,
        ask_order: LimitOrder | None,
    ) -> bool:
        """判断当前 batch 开始前是否需要刷新背景流动性。"""

        if bid_order is None or ask_order is None:
            return True

        tick = max(self.config.price_tick, 0.01)
        if (
            self.batch_clock - self.last_refresh_clock
            >= self.REFRESH_INTERVAL_BATCHES
        ):
            return True
        if abs(anchor_price - self.anchor_price) >= (
            tick * self.REPRICE_THRESHOLD_TICKS
        ):
            return True
        if bid_order.limit_price >= anchor_price or ask_order.limit_price <= anchor_price:
            return True
        return False

    def _replace(
        self,
        step: int,
        batch: int,
        reference_price: float,
    ) -> None:
        """按给定锚点整组替换背景买卖单。"""

        self.order_book.cancel_agent_order(self.bid_agent_id)
        self.order_book.cancel_agent_order(self.ask_agent_id)

        liquidity_depth = max(2, self.config.num_agents // 3)
        spread_ratio = 0.003 + self.rng.uniform(0.0, 0.002)
        tick = max(self.config.price_tick, 0.01)
        strict_bid_cap = max(
            math.floor((reference_price - 1e-9) / tick) * tick,
            tick,
        )
        strict_ask_floor = max(
            math.ceil((reference_price + 1e-9) / tick) * tick,
            tick,
        )
        bid_limit_price = align_limit_price(
            reference_price * (1 - spread_ratio),
            TradeAction.BUY,
            self.config.price_tick,
        )
        ask_limit_price = align_limit_price(
            reference_price * (1 + spread_ratio),
            TradeAction.SELL,
            self.config.price_tick,
        )
        bid_limit_price = min(bid_limit_price, strict_bid_cap)
        ask_limit_price = max(ask_limit_price, strict_ask_floor)

        bid_sequence = self.next_sequence()
        ask_sequence = self.next_sequence()

        bid_order = LimitOrder(
            order_id=f"{self.bid_agent_id}_{step}_{batch}_{bid_sequence}",
            agent_id=self.bid_agent_id,
            action=TradeAction.BUY,
            quantity=liquidity_depth,
            limit_price=bid_limit_price,
            submitted_step=step,
            submitted_batch=batch,
            sequence=bid_sequence,
            reason="background liquidity",
            signal_strength=0.0,
        )
        ask_order = LimitOrder(
            order_id=f"{self.ask_agent_id}_{step}_{batch}_{ask_sequence}",
            agent_id=self.ask_agent_id,
            action=TradeAction.SELL,
            quantity=liquidity_depth,
            limit_price=ask_limit_price,
            submitted_step=step,
            submitted_batch=batch,
            sequence=ask_sequence,
            reason="background liquidity",
            signal_strength=0.0,
        )

        self.order_book.add_order(bid_order)
        self.order_book.add_order(ask_order)
        self.anchor_price = reference_price
        self.last_refresh_clock = self.batch_clock
