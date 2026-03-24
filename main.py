from config import build_experiment_config
from market import SimpleMarket
from models import DecisionMode, StrategyType
from visualization import save_experiment_report


def main() -> None:
    config = build_experiment_config(
        strategy_type=StrategyType.FUNDAMENTAL,
        decision_mode=DecisionMode.RULE_BASED,
    )

    market = SimpleMarket(config)
    market.run_simulation()
    summary = market.get_summary()
    output_dir = save_experiment_report(summary)

    print("=== Mini Simulation Summary ===")
    print(f"strategy_type: {summary['strategy_type']}")
    print(f"decision_mode: {summary['decision_mode']}")
    print(f"num_agents: {summary['num_agents']}")
    print(f"num_steps: {summary['num_steps']}")
    print(f"final_price: {summary['final_price']:.4f}")
    print(f"price_history: {summary['price_history']}")
    print(f"volume_history: {summary['volume_history']}")
    print(f"net_demand_history: {summary['net_demand_history']}")
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
