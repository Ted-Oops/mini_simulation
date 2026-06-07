import argparse

from config import build_experiment_config
from llm_client import get_default_llm_client
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

    if config.decision_mode != DecisionMode.RULE_BASED:
        llm_client = get_default_llm_client()
        if llm_client is not None and llm_client.is_configured:
            print(
                "llm_client: enabled "
                f"({llm_client.config.model} @ {llm_client.config.base_url})"
            )
        else:
            print("llm_client: missing API key; using local rule fallback")

    market = OrderBookMarket(config)
    if config.decision_mode == DecisionMode.RULE_BASED:
        market.run_simulation()
    else:
        estimated_requests = config.num_steps * len(market.agents)
        batches_per_step = (
            len(market.agents) + max(config.batch_size, 1) - 1
        ) // max(config.batch_size, 1)
        print(
            "llm_progress: "
            f"running about {estimated_requests} LLM decisions in "
            f"{batches_per_step} batch-parallel groups per step "
            f"({len(market.agents)} agents × {config.num_steps} steps)",
            flush=True,
        )
        for _ in range(config.num_steps):
            market.run_step()
            latest_volume = market.market_state.volume_history[-1]
            latest_trades = market.trade_count_history[-1]
            print(
                "llm_progress: "
                f"step {market.market_state.step}/{config.num_steps}, "
                f"price={market.market_state.price:.4f}, "
                f"volume={latest_volume}, trades={latest_trades}",
                flush=True,
            )
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
