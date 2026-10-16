import random
from concurrent.futures import ThreadPoolExecutor

from agents import BaseAgent
from models import AgentObservation, DecisionMode, OrderDecision


def _decide_agent(agent: BaseAgent, observation: AgentObservation) -> OrderDecision:
    return agent.decide(observation)


def build_agent_batches(
    agents: list[BaseAgent],
    rng: random.Random,
    batch_size: int,
) -> list[list[BaseAgent]]:
    """随机打散 agent，并按固定 batch 大小切分。"""

    shuffled_agents = list(agents)
    rng.shuffle(shuffled_agents)

    normalized_batch_size = max(1, min(batch_size, len(shuffled_agents)))
    return [
        shuffled_agents[index : index + normalized_batch_size]
        for index in range(0, len(shuffled_agents), normalized_batch_size)
    ]


def decide_batch(
    decision_mode: DecisionMode,
    batch_size: int,
    batch_agents: list[BaseAgent],
    observations: list[AgentObservation],
) -> list[OrderDecision]:
    """为同一个 batch 的 agent 生成决策。"""

    if decision_mode == DecisionMode.RULE_BASED or len(batch_agents) <= 1:
        return [
            agent.decide(observation)
            for agent, observation in zip(batch_agents, observations)
        ]

    max_workers = min(len(batch_agents), max(1, batch_size))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_decide_agent, batch_agents, observations))
