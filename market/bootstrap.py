import math
import random

from agents import BaseAgent
from models import ExperimentConfig, MarketState, TradeAction


def bootstrap_market_context(
    config: ExperimentConfig,
    rng: random.Random,
    market_state: MarketState,
    agents: list[BaseAgent],
) -> tuple[dict[str, TradeAction], dict[str, int]] | None:
    """构造正式交易前的 warm-up 历史，并返回初始化后的同伴状态。"""

    if config.bootstrap_steps <= 0:
        return None

    bootstrap_returns = generate_bootstrap_returns(config.bootstrap_steps, rng)
    bootstrap_prices = build_bootstrap_prices(config, rng, bootstrap_returns)
    bootstrap_prices = shape_bootstrap_prices(config, rng, bootstrap_prices)
    bootstrap_returns = returns_from_prices(bootstrap_prices)
    bootstrap_net_demand = generate_bootstrap_net_demand(
        config,
        rng,
        bootstrap_returns,
    )
    bootstrap_volume = generate_bootstrap_volume(config, rng, bootstrap_net_demand)

    market_state.price_history = bootstrap_prices
    market_state.mark_price_history = list(bootstrap_prices)
    market_state.return_history = bootstrap_returns
    market_state.net_demand_history = bootstrap_net_demand
    market_state.volume_history = bootstrap_volume
    market_state.price = bootstrap_prices[-1]
    market_state.mark_price = bootstrap_prices[-1]

    return seed_peer_state_from_bootstrap(rng, agents, bootstrap_returns)


def generate_bootstrap_returns(num_steps: int, rng: random.Random) -> list[float]:
    """生成 warm-up 收益率序列。"""

    bootstrap_returns: list[float] = []
    previous_return = 0.0

    for step in range(num_steps):
        cyclical_component = 0.009 * math.sin(
            2 * math.pi * (step + 1) / 6.5 + 0.55
        ) + 0.006 * math.cos(2 * math.pi * (step + 1) / 11.0 + 1.10)
        persistence_component = 0.28 * previous_return
        noise_component = rng.uniform(-0.012, 0.012)
        burst_component = 0.0

        if step % 7 in (2, 5):
            burst_component = rng.choice([-1.0, 1.0]) * rng.uniform(0.004, 0.014)

        period_return = (
            cyclical_component
            + persistence_component
            + noise_component
            + burst_component
        )
        period_return = max(min(period_return, 0.032), -0.032)

        bootstrap_returns.append(period_return)
        previous_return = period_return

    return bootstrap_returns


def build_bootstrap_prices(
    config: ExperimentConfig,
    rng: random.Random,
    bootstrap_returns: list[float],
) -> list[float]:
    """把 warm-up 收益率转换成价格路径。"""

    start_price = config.initial_price * (0.94 + 0.05 * rng.random())
    bootstrap_prices = [start_price]

    for period_return in bootstrap_returns:
        next_price = max(bootstrap_prices[-1] * (1 + period_return), 0.01)
        bootstrap_prices.append(next_price)

    scaling_factor = config.initial_price / bootstrap_prices[-1]
    return [max(price * scaling_factor, 0.01) for price in bootstrap_prices]


def generate_bootstrap_net_demand(
    config: ExperimentConfig,
    rng: random.Random,
    bootstrap_returns: list[float],
) -> list[float]:
    """根据 warm-up 收益率构造一条配套的净需求历史。"""

    bootstrap_net_demand: list[float] = []
    max_abs_demand = max(2, config.num_agents - 1)

    for period_return in bootstrap_returns:
        noisy_signal = (period_return / 0.012) * (config.num_agents / 3)
        noisy_signal += rng.uniform(-1.5, 1.5)
        signed_demand = round(noisy_signal)
        signed_demand = max(min(signed_demand, max_abs_demand), -max_abs_demand)

        if signed_demand == 0 and abs(period_return) > 0.008:
            signed_demand = 1 if period_return > 0 else -1

        bootstrap_net_demand.append(signed_demand)

    return bootstrap_net_demand


def generate_bootstrap_volume(
    config: ExperimentConfig,
    rng: random.Random,
    bootstrap_net_demand: list[float],
) -> list[int]:
    """为 warm-up 历史生成一个量级合理的成交量序列。"""

    bootstrap_volume: list[int] = []
    base_volume = max(3, config.num_agents // 2)

    for signed_demand in bootstrap_net_demand:
        extra_turnover = rng.randint(0, base_volume + 2)
        bootstrap_volume.append(int(abs(signed_demand) + base_volume + extra_turnover))

    return bootstrap_volume


def seed_peer_state_from_bootstrap(
    rng: random.Random,
    agents: list[BaseAgent],
    bootstrap_returns: list[float],
) -> tuple[dict[str, TradeAction], dict[str, int]]:
    """根据 warm-up 尾部状态，初始化上一轮同伴行为特征。"""

    if not bootstrap_returns:
        return (
            {agent.agent_id: TradeAction.HOLD for agent in agents},
            {agent.agent_id: 0 for agent in agents},
        )

    recent_mean_return = sum(bootstrap_returns[-3:]) / min(3, len(bootstrap_returns))
    current_sentiment = 0.6 * bootstrap_returns[-1] + 0.4 * recent_mean_return

    if current_sentiment > 0.008:
        buy_probability, sell_probability = 0.45, 0.20
    elif current_sentiment < -0.008:
        buy_probability, sell_probability = 0.20, 0.45
    else:
        buy_probability, sell_probability = 0.34, 0.30

    seeded_actions: dict[str, TradeAction] = {}
    seeded_quantities: dict[str, int] = {}

    for agent in agents:
        draw = rng.random()
        if draw < buy_probability:
            action = TradeAction.BUY
            signed_quantity = 1
        elif draw < buy_probability + sell_probability:
            action = TradeAction.SELL
            signed_quantity = -1
        else:
            action = TradeAction.HOLD
            signed_quantity = 0

        seeded_actions[agent.agent_id] = action
        seeded_quantities[agent.agent_id] = signed_quantity

    unique_actions = set(seeded_actions.values())
    if len(unique_actions) == 1 and len(agents) >= 2:
        first_agent_id = agents[0].agent_id
        second_agent_id = agents[1].agent_id
        seeded_actions[first_agent_id] = TradeAction.BUY
        seeded_quantities[first_agent_id] = 1
        seeded_actions[second_agent_id] = TradeAction.SELL
        seeded_quantities[second_agent_id] = -1

    return seeded_actions, seeded_quantities


def seed_initial_inventory(
    config: ExperimentConfig,
    market_state: MarketState,
    agents: list[BaseAgent],
) -> None:
    """给每个 agent 分配一小笔等值初始持仓。"""

    initial_shares = max(config.initial_shares_per_agent, 0)
    if initial_shares == 0:
        return

    reference_price = max(market_state.price, config.price_tick)
    for agent in agents:
        affordable_shares = min(initial_shares, int(agent.state.cash // reference_price))
        if affordable_shares <= 0:
            continue

        agent.state.cash -= affordable_shares * reference_price
        agent.state.shares = affordable_shares
        agent.state.avg_cost = reference_price


def shape_bootstrap_prices(
    config: ExperimentConfig,
    rng: random.Random,
    bootstrap_prices: list[float],
) -> list[float]:
    """对 warm-up 尾部做轻微塑形，让历史更有结构感。"""

    if len(bootstrap_prices) < 6:
        return bootstrap_prices

    final_price = config.initial_price
    tail_template = [
        1.145 + rng.uniform(-0.010, 0.010),
        1.085 + rng.uniform(-0.008, 0.008),
        0.945 + rng.uniform(-0.008, 0.008),
        0.972 + rng.uniform(-0.006, 0.006),
        1.000,
    ]
    shaped_tail = [max(final_price * multiplier, 0.01) for multiplier in tail_template]
    return bootstrap_prices[:-5] + shaped_tail


def returns_from_prices(bootstrap_prices: list[float]) -> list[float]:
    """把价格序列转换成收益率序列。"""

    bootstrap_returns: list[float] = []
    for index in range(len(bootstrap_prices) - 1):
        current_price = bootstrap_prices[index]
        next_price = bootstrap_prices[index + 1]
        if current_price <= 0:
            bootstrap_returns.append(0.0)
        else:
            bootstrap_returns.append((next_price - current_price) / current_price)
    return bootstrap_returns
