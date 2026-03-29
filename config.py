from models import ExperimentConfig, StrategyType, DecisionMode


DEFAULT_NUM_AGENTS = 12
DEFAULT_NUM_STEPS = 50
DEFAULT_BOOTSTRAP_STEPS = 24
DEFAULT_INITIAL_PRICE = 100.0
DEFAULT_INITIAL_CASH = 10000.0
DEFAULT_INITIAL_SHARES_PER_AGENT = 5
DEFAULT_BATCH_SIZE = 3
DEFAULT_PRICE_IMPACT = 0.01
DEFAULT_PRICE_TICK = 0.1
DEFAULT_MAX_ORDER_AGE_STEPS = 3
DEFAULT_MAX_PRICE_OFFSET_RATIO = 0.03
DEFAULT_SEED = 42


def build_experiment_config(
    strategy_type: StrategyType, decision_mode: DecisionMode
) -> ExperimentConfig:
    return ExperimentConfig(
        strategy_type=strategy_type,
        decision_mode=decision_mode,
        num_agents=DEFAULT_NUM_AGENTS,
        num_steps=DEFAULT_NUM_STEPS,
        bootstrap_steps=DEFAULT_BOOTSTRAP_STEPS,
        initial_price=DEFAULT_INITIAL_PRICE,
        initial_cash=DEFAULT_INITIAL_CASH,
        initial_shares_per_agent=DEFAULT_INITIAL_SHARES_PER_AGENT,
        batch_size=DEFAULT_BATCH_SIZE,
        price_impact=DEFAULT_PRICE_IMPACT,
        price_tick=DEFAULT_PRICE_TICK,
        max_order_age_steps=DEFAULT_MAX_ORDER_AGE_STEPS,
        max_price_offset_ratio=DEFAULT_MAX_PRICE_OFFSET_RATIO,
        seed=DEFAULT_SEED,
    )
