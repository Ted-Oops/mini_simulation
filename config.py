from models import ExperimentConfig, StrategyType, DecisionMode


DEFAULT_NUM_AGENTS = 10
DEFAULT_NUM_STEPS = 50
DEFAULT_BOOTSTRAP_STEPS = 24
DEFAULT_INITIAL_PRICE = 100.0
DEFAULT_INITIAL_CASH = 10000.0
DEFAULT_LOT_SIZE = 1
DEFAULT_PRICE_IMPACT = 0.01
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
        lot_size=DEFAULT_LOT_SIZE,
        price_impact=DEFAULT_PRICE_IMPACT,
        seed=DEFAULT_SEED,
        enable_shock=False,
    )


def build_all_experiment_configs() -> list[ExperimentConfig]:
    configs = []

    for strategy_type in StrategyType:
        for decision_mode in DecisionMode:
            configs.append(build_experiment_config(strategy_type, decision_mode))

    return configs
