import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


STRATEGY_ORDER = ["fundamental", "momentum", "speculator"]
DECISION_ORDER = ["rule_based", "half_rule_based", "open_ended"]


def _grid_position(strategy_type: str, decision_mode: str) -> tuple[int, int]:
    row = STRATEGY_ORDER.index(strategy_type) + 1 if strategy_type in STRATEGY_ORDER else -1
    col = DECISION_ORDER.index(decision_mode) + 1 if decision_mode in DECISION_ORDER else -1
    return row, col


def _ensure_output_dir(output_root: str | Path, experiment_label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"{experiment_label}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _build_experiment_label(summary: dict) -> str:
    strategy_type = summary.get("strategy_type", "unknown")
    decision_mode = summary.get("decision_mode", "unknown")
    return f"{strategy_type}__{decision_mode}"


def _compute_agent_wealth(agent_info: dict, final_price: float) -> float:
    cash = agent_info.get("cash", 0.0)
    shares = agent_info.get("shares", 0)
    return cash + shares * final_price


def save_experiment_report(summary: dict, output_root: str | Path = "artifacts") -> Path:
    experiment_label = _build_experiment_label(summary)
    output_dir = _ensure_output_dir(output_root, experiment_label)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    strategy_type = summary.get("strategy_type", "unknown")
    decision_mode = summary.get("decision_mode", "unknown")
    row, col = _grid_position(strategy_type, decision_mode)

    price_history = summary.get("price_history", [])
    volume_history = summary.get("volume_history", [])
    net_demand_history = summary.get("net_demand_history", [])
    fundamental_value_history = summary.get("fundamental_value_history", [])

    buy_count_history = summary.get("buy_count_history", [])
    sell_count_history = summary.get("sell_count_history", [])
    hold_count_history = summary.get("hold_count_history", [])
    rejected_count_history = summary.get("rejected_count_history", [])

    agents = summary.get("agents", [])
    final_price = summary.get("final_price", price_history[-1] if price_history else 0.0)

    if not fundamental_value_history and price_history:
        initial_fundamental = summary.get("initial_fundamental_value", final_price)
        fundamental_value_history = [initial_fundamental] * len(price_history)

    wealth_pairs = []
    for agent_info in agents:
        wealth = _compute_agent_wealth(agent_info, final_price)
        wealth_pairs.append((agent_info.get("agent_id", "unknown"), wealth))
    wealth_pairs.sort(key=lambda x: x[1], reverse=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    title = (
        f"Mini Simulation Report: {strategy_type} × {decision_mode} | "
        f"Grid=({row}, {col}) | Steps={summary.get('num_steps', 0)} | "
        f"Agents={summary.get('num_agents', 0)}"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax1 = axes[0, 0]
    if price_history:
        x_price = list(range(len(price_history)))
        ax1.plot(x_price, price_history, color="#1f77b4", linewidth=2.2, label="Market Price")
    if fundamental_value_history:
        x_fv = list(range(len(fundamental_value_history)))
        ax1.plot(
            x_fv,
            fundamental_value_history,
            color="#ff7f0e",
            linewidth=2.0,
            linestyle="--",
            label="Fundamental Value",
        )
    ax1.set_title("Price Path")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Price")
    ax1.legend()

    ax2 = axes[0, 1]
    if net_demand_history:
        x_demand = list(range(1, len(net_demand_history) + 1))
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in net_demand_history]
        ax2.bar(x_demand, net_demand_history, color=colors, alpha=0.8, label="Signed Net Demand")
    if volume_history:
        x_volume = list(range(1, len(volume_history) + 1))
        ax2.plot(x_volume, volume_history, color="#4c4c4c", linewidth=1.8, label="Absolute Volume")
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_title("Demand and Volume")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Shares")
    ax2.legend()

    ax3 = axes[1, 0]
    has_action_data = any(
        [buy_count_history, sell_count_history, hold_count_history, rejected_count_history]
    )
    if has_action_data:
        x_counts = list(
            range(
                1,
                max(
                    len(buy_count_history),
                    len(sell_count_history),
                    len(hold_count_history),
                    len(rejected_count_history),
                ) + 1,
            )
        )

        if buy_count_history:
            ax3.plot(
                x_counts[: len(buy_count_history)],
                buy_count_history,
                color="#2ca02c",
                linewidth=2.0,
                label="Buy Count",
            )
        if sell_count_history:
            ax3.plot(
                x_counts[: len(sell_count_history)],
                sell_count_history,
                color="#d62728",
                linewidth=2.0,
                label="Sell Count",
            )
        if hold_count_history:
            ax3.plot(
                x_counts[: len(hold_count_history)],
                hold_count_history,
                color="#7f7f7f",
                linewidth=2.0,
                label="Hold Count",
            )
        if rejected_count_history:
            ax3.plot(
                x_counts[: len(rejected_count_history)],
                rejected_count_history,
                color="#9467bd",
                linewidth=2.0,
                linestyle=":",
                label="Rejected Count",
            )
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No action-count data", ha="center", va="center", fontsize=12)
    ax3.set_title("Agent Actions by Step")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Count")

    ax4 = axes[1, 1]
    if wealth_pairs:
        agent_ids = [item[0] for item in wealth_pairs]
        wealth_values = [item[1] for item in wealth_pairs]
        ax4.bar(agent_ids, wealth_values, color="#17becf", alpha=0.85)
        ax4.tick_params(axis="x", rotation=45)
    else:
        ax4.text(0.5, 0.5, "No agent wealth data", ha="center", va="center", fontsize=12)
    ax4.set_title("Final Agent Wealth")
    ax4.set_xlabel("Agent")
    ax4.set_ylabel("Wealth")

    if price_history:
        initial_price = price_history[0]
        final_return = (final_price - initial_price) / initial_price if initial_price != 0 else 0.0
    else:
        final_return = 0.0

    footer_text = (
        f"Final Price={final_price:.4f} | "
        f"Final Return={final_return:.2%} | "
        f"Experiment={experiment_label}"
    )
    fig.text(0.5, 0.01, footer_text, ha="center", fontsize=11)

    figure_path = output_dir / "report.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return output_dir
