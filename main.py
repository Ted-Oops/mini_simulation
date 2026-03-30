import argparse

from config import build_experiment_config
from market import OrderBookMarket
from models import DecisionMode, StrategyType
from visualization import save_experiment_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a mini market simulation experiment."
    )
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in StrategyType],
        default=StrategyType.FUNDAMENTAL.value,
        help="Agent strategy type.",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in DecisionMode],
        default=DecisionMode.RULE_BASED.value,
        help="Decision mode.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the number of simulation steps.",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=None,
        help="Override the number of agents.",
    )
    parser.add_argument(
        "--bootstrap-steps",
        type=int,
        default=None,
        help="Override the number of warm-up history steps before step 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = build_experiment_config(
        strategy_type=StrategyType(args.strategy),
        decision_mode=DecisionMode(args.mode),
    )

    if args.steps is not None:
        config.num_steps = args.steps
    if args.agents is not None:
        config.num_agents = args.agents
    if args.bootstrap_steps is not None:
        config.bootstrap_steps = args.bootstrap_steps

    market = OrderBookMarket(config)
    market.run_simulation()
    summary = market.get_summary()
    output_dir = save_experiment_report(summary)

    print("=== Mini Simulation Summary ===")
    print(f"strategy_type: {summary['strategy_type']}")
    print(f"decision_mode: {summary['decision_mode']}")
    print(f"num_agents: {summary['num_agents']}")
    print(f"num_steps: {summary['num_steps']}")
    print(f"bootstrap_steps: {summary['bootstrap_steps']}")
    print(f"batch_size: {summary['batch_size']}")
    print(f"initial_shares_per_agent: {summary['initial_shares_per_agent']}")
    print(f"final_price: {summary['final_price']:.4f}")
    print(f"final_mark_price: {summary['final_mark_price']:.4f}")
    print(f"price_history: {summary['price_history']}")
    print(f"volume_history: {summary['volume_history']}")
    print(f"net_demand_history: {summary['net_demand_history']}")
    print(f"trade_count_history: {summary['trade_count_history']}")
    print()

    print("=== Agent States ===")
    for agent_info in summary["agents"]:
        print(
            f"{agent_info['agent_id']}: "
            f"cash={agent_info['cash']:.2f}, "
            f"shares={agent_info['shares']}, "
            f"avg_cost={agent_info['avg_cost']:.2f}"
        )

    print()
    print(f"report_saved_to: {output_dir}")


if __name__ == "__main__":
    main()
