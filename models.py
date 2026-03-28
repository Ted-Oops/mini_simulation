from dataclasses import dataclass, field
from enum import Enum


class StrategyType(str, Enum):
    FUNDAMENTAL = "fundamental"
    MOMENTUM = "momentum"
    SPECULATOR = "speculator"

class DecisionMode(str, Enum):
    RULE_BASED = "rule_based"
    HALF_RULE_BASED = "half_rule_based"
    OPEN_ENDED = "open_ended"

class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class OrderDecision:
    agent_id: str
    action: TradeAction
    quantity: int
    limit_price: float | None = None
    reason: str = ""
    signal_strength: float = 0.0

@dataclass
class AgentState:
    agent_id: str
    cash: float
    shares: int
    avg_cost: float

@dataclass
class MarketState:
    step: int
    price: float
    price_history: list[float] = field(default_factory=list)
    return_history: list[float] = field(default_factory=list)
    volume_history: list[int] = field(default_factory=list)
    net_demand_history: list[float] = field(default_factory=list)
    shock: float = 0.0

@dataclass
class AgentObservation:
    step: int
    price: float
    sue_signal: float
    momentum_1: float
    momentum_3: float
    reversal_score: float
    volatility: float
    net_demand: float
    peer_buy_ratio: float
    peer_sell_ratio: float
    peer_hold_ratio: float
    peer_net_demand: float
    shock: float

@dataclass
class ExperimentConfig:
    strategy_type: StrategyType
    decision_mode: DecisionMode
    num_agents: int
    num_steps: int
    bootstrap_steps: int
    initial_price: float
    initial_cash: float
    lot_size: int = 1
    price_impact: float = 0.01
    seed: int = 42
    enable_shock: bool = False
