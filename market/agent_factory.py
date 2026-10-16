from agents import BaseAgent, FundamentalAgent, MomentumAgent, SpeculatorAgent
from models import DecisionMode, StrategyType


def create_agents(
    strategy_type: StrategyType,
    decision_mode: DecisionMode,
    num_agents: int,
    initial_cash: float,
) -> list[BaseAgent]:
    """根据实验配置实例化一组同策略 agent。"""

    agent_prefix = strategy_type.value

    if strategy_type == StrategyType.FUNDAMENTAL:
        agent_cls = FundamentalAgent
    elif strategy_type == StrategyType.MOMENTUM:
        agent_cls = MomentumAgent
    elif strategy_type == StrategyType.SPECULATOR:
        agent_cls = SpeculatorAgent
    else:
        raise NotImplementedError(f"Strategy type {strategy_type} is not implemented")

    return [
        agent_cls(
            agent_id=f"{agent_prefix}_{i + 1}",
            decision_mode=decision_mode,
            initial_cash=initial_cash,
        )
        for i in range(num_agents)
    ]
