import math
import random

from agents import BaseAgent
from market.order_book import OrderBook
from models import (
    AgentObservation,
    ExperimentConfig,
    LimitOrder,
    OrderDecision,
    StrategyType,
    TradeAction,
    TradeExecution,
)


def align_limit_price(
    raw_price: float,
    action: TradeAction,
    price_tick: float,
) -> float:
    """把原始报价吸附到 tick 网格上。"""

    tick = max(price_tick, 0.01)
    if action == TradeAction.BUY:
        tick_count = math.ceil(raw_price / tick - 1e-9)
    else:
        tick_count = math.floor(raw_price / tick + 1e-9)
    return max(tick_count * tick, tick)


def derive_limit_price(
    *,
    config: ExperimentConfig,
    rng: random.Random,
    order_book: OrderBook,
    agent: BaseAgent,
    observation: AgentObservation,
    decision: OrderDecision,
) -> float:
    """把 agent 决策映射成最终挂单价格。"""

    reference_price = (
        decision.limit_price
        if decision.limit_price is not None and decision.limit_price > 0
        else observation.price
    )
    signal_strength = max(0.0, min(decision.signal_strength, 1.0))
    best_bid = order_book.best_bid()
    best_ask = order_book.best_ask()
    strategy_cross_bonus = {
        StrategyType.FUNDAMENTAL: -0.05,
        StrategyType.MOMENTUM: 0.00,
        StrategyType.SPECULATOR: 0.12,
    }[agent.strategy_type]
    cross_probability = min(
        max(0.32 + 0.48 * signal_strength + strategy_cross_bonus, 0.18),
        0.90,
    )
    base_offset = max(config.price_impact * 0.50, 0.002)
    aggressive_offset = min(
        base_offset * (0.8 + 1.6 * signal_strength) + rng.uniform(0.0, 0.002),
        config.max_price_offset_ratio,
    )
    passive_offset = min(
        base_offset * (0.35 + 0.85 * (1.0 - 0.5 * signal_strength))
        + rng.uniform(0.0, 0.0015),
        config.max_price_offset_ratio,
    )

    if decision.action == TradeAction.BUY:
        if rng.random() < cross_probability:
            anchor_price = max(reference_price, best_ask or reference_price)
            raw_price = anchor_price * (1 + aggressive_offset)
        else:
            anchor_price = min(reference_price, best_bid or reference_price)
            raw_price = anchor_price * (1 - passive_offset)

        if decision.quantity > 0:
            max_affordable_price = agent.state.cash / decision.quantity
            raw_price = min(raw_price, max_affordable_price)

        return align_limit_price(raw_price, TradeAction.BUY, config.price_tick)

    anchor_price = min(reference_price, best_bid or reference_price)
    if rng.random() < cross_probability:
        raw_price = anchor_price * (1 - aggressive_offset)
    else:
        anchor_price = max(reference_price, best_ask or reference_price)
        raw_price = anchor_price * (1 + passive_offset)

    return align_limit_price(raw_price, TradeAction.SELL, config.price_tick)


def create_limit_order(
    *,
    config: ExperimentConfig,
    rng: random.Random,
    order_book: OrderBook,
    order_sequence: int,
    agent: BaseAgent,
    observation: AgentObservation,
    decision: OrderDecision,
    step: int,
    batch: int,
) -> tuple[LimitOrder | None, int]:
    """把决策对象转换成真正可入簿的限价单。"""

    if decision.action == TradeAction.HOLD or decision.quantity <= 0:
        return None, order_sequence

    limit_price = derive_limit_price(
        config=config,
        rng=rng,
        order_book=order_book,
        agent=agent,
        observation=observation,
        decision=decision,
    )
    if limit_price <= 0:
        return None, order_sequence

    if decision.action == TradeAction.BUY:
        max_affordable_quantity = int(agent.state.cash // limit_price)
        quantity = min(decision.quantity, max_affordable_quantity)
    else:
        quantity = min(decision.quantity, agent.state.shares)

    if quantity <= 0:
        return None, order_sequence

    next_order_sequence = order_sequence + 1
    return (
        LimitOrder(
            order_id=f"{agent.agent_id}_order_{next_order_sequence}",
            agent_id=agent.agent_id,
            action=decision.action,
            quantity=quantity,
            limit_price=limit_price,
            submitted_step=step,
            submitted_batch=batch,
            sequence=next_order_sequence,
            reason=decision.reason,
            signal_strength=decision.signal_strength,
        ),
        next_order_sequence,
    )


def signed_quantity(action: TradeAction, quantity: int) -> int:
    """把买卖方向转换成带符号的数量。"""

    if action == TradeAction.BUY:
        return quantity
    if action == TradeAction.SELL:
        return -quantity
    return 0


def apply_execution(
    agent: BaseAgent,
    action: TradeAction,
    quantity: int,
    execution_price: float,
) -> None:
    """把一笔成交真正记入 agent 账户。"""

    execution_decision = OrderDecision(
        agent_id=agent.agent_id,
        action=action,
        quantity=quantity,
        limit_price=execution_price,
        reason="matched in the order book",
        signal_strength=0.0,
    )
    agent.apply_trade(execution_decision, execution_price)


def settle_trade(
    agent_lookup: dict[str, BaseAgent],
    trade: TradeExecution,
) -> None:
    """对一笔成交执行资金和持仓结算。"""

    buyer = agent_lookup.get(trade.buyer_id)
    seller = agent_lookup.get(trade.seller_id)

    if buyer is not None:
        apply_execution(
            buyer,
            TradeAction.BUY,
            trade.quantity,
            trade.price,
        )
    if seller is not None:
        apply_execution(
            seller,
            TradeAction.SELL,
            trade.quantity,
            trade.price,
        )
