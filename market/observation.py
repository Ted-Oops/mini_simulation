import math

from agents import BaseAgent
from models import AgentObservation, MarketState, StrategyType, TradeAction


def generate_sue_signal(step: int) -> float:
    """生成供基本面 agent 使用的模拟 SUE 风格信号。"""

    return (
        0.80 * math.sin(2 * math.pi * step / 12)
        + 0.45 * math.sin(2 * math.pi * step / 5 + math.pi / 6)
        + 0.25 * math.cos(2 * math.pi * step / 21 + math.pi / 4)
    )


def compute_reversal_score(market_state: MarketState) -> float:
    """根据最近价格窗口计算一个简单反转分数。"""

    prices = market_state.price_history
    if len(prices) < 3:
        return 0.0

    window = prices[-5:] if len(prices) >= 5 else prices
    mean_price = sum(window) / len(window)
    if mean_price == 0:
        return 0.0
    return (mean_price - market_state.price) / mean_price


def get_peer_action_features(
    agent_id: str,
    last_actions_by_agent: dict[str, TradeAction],
    last_signed_quantity_by_agent: dict[str, int],
) -> tuple[float, float, float, float]:
    """提取某个 agent 视角下的同伴行为特征。"""

    peer_items = [
        (peer_id, action)
        for peer_id, action in last_actions_by_agent.items()
        if peer_id != agent_id
    ]
    if not peer_items:
        return 0.0, 0.0, 1.0, 0.0

    peer_count = len(peer_items)
    buy_ratio = sum(action == TradeAction.BUY for _, action in peer_items) / peer_count
    sell_ratio = (
        sum(action == TradeAction.SELL for _, action in peer_items) / peer_count
    )
    hold_ratio = (
        sum(action == TradeAction.HOLD for _, action in peer_items) / peer_count
    )
    peer_net_demand = sum(
        last_signed_quantity_by_agent.get(peer_id, 0) for peer_id, _ in peer_items
    )
    return buy_ratio, sell_ratio, hold_ratio, peer_net_demand


def build_observation(
    market_state: MarketState,
    agent: BaseAgent,
    last_actions_by_agent: dict[str, TradeAction],
    last_signed_quantity_by_agent: dict[str, int],
) -> AgentObservation:
    """为单个 agent 构造本轮观测。"""

    sue_signal = generate_sue_signal(market_state.step)
    reversal_score = compute_reversal_score(market_state)
    if agent.strategy_type == StrategyType.SPECULATOR:
        (
            peer_buy_ratio,
            peer_sell_ratio,
            peer_hold_ratio,
            peer_net_demand,
        ) = get_peer_action_features(
            agent.agent_id,
            last_actions_by_agent,
            last_signed_quantity_by_agent,
        )
    else:
        peer_buy_ratio = 0.0
        peer_sell_ratio = 0.0
        peer_hold_ratio = 0.0
        peer_net_demand = 0.0

    if len(market_state.return_history) >= 1:
        momentum_1 = market_state.return_history[-1]
    else:
        momentum_1 = 0.0

    if len(market_state.return_history) >= 3:
        momentum_3 = sum(market_state.return_history[-3:]) / 3
    elif len(market_state.return_history) > 0:
        momentum_3 = sum(market_state.return_history) / len(
            market_state.return_history
        )
    else:
        momentum_3 = 0.0

    if len(market_state.return_history) >= 2:
        mean_return = sum(market_state.return_history) / len(
            market_state.return_history
        )
        squared_diffs = [
            (ret - mean_return) ** 2 for ret in market_state.return_history
        ]
        volatility = (sum(squared_diffs) / len(squared_diffs)) ** 0.5
    else:
        volatility = 0.0

    if len(market_state.net_demand_history) >= 1:
        net_demand = market_state.net_demand_history[-1]
    else:
        net_demand = 0.0

    return AgentObservation(
        step=market_state.step,
        price=market_state.price,
        sue_signal=sue_signal,
        momentum_1=momentum_1,
        momentum_3=momentum_3,
        reversal_score=reversal_score,
        volatility=volatility,
        net_demand=net_demand,
        peer_buy_ratio=peer_buy_ratio,
        peer_sell_ratio=peer_sell_ratio,
        peer_hold_ratio=peer_hold_ratio,
        peer_net_demand=peer_net_demand,
        shock=market_state.shock,
    )
