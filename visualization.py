import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


STRATEGY_ORDER = ["fundamental", "momentum", "speculator"]
DECISION_ORDER = ["rule_based", "half_rule_based", "open_ended"]

FIGURE_BG = "#f7f4ed"
PANEL_BG = "#fffdf8"
GRID_COLOR = "#d8d0c3"
TEXT_COLOR = "#2a2a2a"
PRICE_COLOR = "#0f766e"
PRICE_FILL = "#99f6e4"
DEMAND_POSITIVE = "#d97706"
DEMAND_NEGATIVE = "#b91c1c"
VOLUME_COLOR = "#334155"
BUY_COLOR = "#0f766e"
SELL_COLOR = "#b45309"
HOLD_COLOR = "#64748b"
BLOCKED_COLOR = "#9a3412"
WEALTH_GRADIENT_START = "#94d2bd"
WEALTH_GRADIENT_END = "#005f73"


def _grid_position(strategy_type: str, decision_mode: str) -> tuple[int, int]:
    row = (
        STRATEGY_ORDER.index(strategy_type) + 1
        if strategy_type in STRATEGY_ORDER
        else -1
    )
    col = (
        DECISION_ORDER.index(decision_mode) + 1
        if decision_mode in DECISION_ORDER
        else -1
    )
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


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID_COLOR, alpha=0.55, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b5afa3")
    ax.spines["bottom"].set_color("#b5afa3")
    ax.tick_params(colors=TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


def _set_dynamic_ylim(
    ax: plt.Axes,
    values: list[float],
    pad_ratio: float,
    min_pad: float,
) -> None:
    if not values:
        return

    value_min = min(values)
    value_max = max(values)
    if math.isclose(value_min, value_max):
        pad = max(abs(value_min) * pad_ratio, min_pad)
    else:
        pad = max((value_max - value_min) * pad_ratio, min_pad)
    ax.set_ylim(value_min - pad, value_max + pad)


def _blend_hex_colors(start_hex: str, end_hex: str, count: int) -> list[tuple[float, float, float]]:
    if count <= 0:
        return []
    if count == 1:
        return [mcolors.to_rgb(end_hex)]

    start = mcolors.to_rgb(start_hex)
    end = mcolors.to_rgb(end_hex)
    colors: list[tuple[float, float, float]] = []
    for index in range(count):
        weight = index / (count - 1)
        colors.append(
            tuple(
                start[channel] * (1 - weight) + end[channel] * weight
                for channel in range(3)
            )
        )
    return colors


def save_experiment_report(
    summary: dict, output_root: str | Path = "artifacts"
) -> Path:
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

    buy_count_history = summary.get("buy_count_history", [])
    sell_count_history = summary.get("sell_count_history", [])
    hold_count_history = summary.get("hold_count_history", [])
    blocked_count_history = summary.get(
        "blocked_count_history",
        summary.get("rejected_count_history", []),
    )

    agents = summary.get("agents", [])
    final_price = summary.get(
        "final_price", price_history[-1] if price_history else 0.0
    )

    wealth_pairs = []
    for agent_info in agents:
        wealth = _compute_agent_wealth(agent_info, final_price)
        wealth_pairs.append((agent_info.get("agent_id", "unknown"), wealth))
    wealth_pairs.sort(key=lambda x: x[1], reverse=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), constrained_layout=True)
    fig.patch.set_facecolor(FIGURE_BG)

    title = (
        f"Mini Simulation Report: {strategy_type} × {decision_mode} | "
        f"Grid=({row}, {col}) | Steps={summary.get('num_steps', 0)} | "
        f"Agents={summary.get('num_agents', 0)} | "
        f"Warm-up={summary.get('bootstrap_steps', 0)}"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", color=TEXT_COLOR)

    ax1 = axes[0, 0]
    _style_axis(ax1)
    if price_history:
        x_price = list(range(len(price_history)))
        baseline = min(price_history)
        ax1.plot(
            x_price,
            price_history,
            color=PRICE_COLOR,
            linewidth=2.6,
            marker="o",
            markersize=4.5,
            markerfacecolor="#ffffff",
            markeredgewidth=1.2,
            label="Last Trade Price",
        )
        ax1.fill_between(
            x_price,
            price_history,
            [baseline] * len(price_history),
            color=PRICE_FILL,
            alpha=0.18,
        )
        _set_dynamic_ylim(ax1, list(price_history), pad_ratio=0.18, min_pad=0.12)
        ax1.legend(frameon=False, loc="upper left")
    else:
        ax1.text(
            0.5, 0.5, "No price data", ha="center", va="center", fontsize=12
        )
    ax1.set_title("Last Trade Price Path", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Price")

    ax2 = axes[0, 1]
    _style_axis(ax2)
    if net_demand_history:
        x_demand = list(range(1, len(net_demand_history) + 1))
        colors = [
            DEMAND_POSITIVE if value >= 0 else DEMAND_NEGATIVE
            for value in net_demand_history
        ]
        ax2.bar(
            x_demand,
            net_demand_history,
            width=0.72,
            color=colors,
            alpha=0.82,
            label="Net Demand",
        )
    ax2.axhline(0, color="#6b7280", linewidth=1.0, alpha=0.8)
    ax2.set_title("Demand and Volume", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Signed Net Demand")

    volume_axis = ax2.twinx()
    volume_axis.set_facecolor("none")
    volume_axis.spines["top"].set_visible(False)
    volume_axis.spines["left"].set_visible(False)
    volume_axis.spines["right"].set_color("#b5afa3")
    volume_axis.tick_params(colors=TEXT_COLOR)
    volume_axis.yaxis.label.set_color(TEXT_COLOR)
    if volume_history:
        x_volume = list(range(1, len(volume_history) + 1))
        volume_axis.plot(
            x_volume,
            volume_history,
            color=VOLUME_COLOR,
            linewidth=2.0,
            marker="o",
            markersize=3.8,
            label="Traded Volume",
        )
    volume_axis.set_ylabel("Traded Volume")

    demand_handles, demand_labels = ax2.get_legend_handles_labels()
    volume_handles, volume_labels = volume_axis.get_legend_handles_labels()
    if demand_handles or volume_handles:
        ax2.legend(
            demand_handles + volume_handles,
            demand_labels + volume_labels,
            frameon=False,
            loc="upper left",
        )

    ax3 = axes[1, 0]
    _style_axis(ax3)
    has_action_data = any(
        [
            buy_count_history,
            sell_count_history,
            hold_count_history,
            blocked_count_history,
        ]
    )
    if has_action_data:
        max_length = max(
            len(buy_count_history),
            len(sell_count_history),
            len(hold_count_history),
            len(blocked_count_history),
        )
        x_counts = list(range(1, max_length + 1))

        if buy_count_history:
            ax3.plot(
                x_counts[: len(buy_count_history)],
                buy_count_history,
                color=BUY_COLOR,
                linewidth=2.1,
                marker="o",
                markersize=3.5,
                label="Buy Orders",
            )
        if sell_count_history:
            ax3.plot(
                x_counts[: len(sell_count_history)],
                sell_count_history,
                color=SELL_COLOR,
                linewidth=2.1,
                marker="o",
                markersize=3.5,
                label="Sell Orders",
            )
        if hold_count_history:
            ax3.plot(
                x_counts[: len(hold_count_history)],
                hold_count_history,
                color=HOLD_COLOR,
                linewidth=2.0,
                marker="o",
                markersize=3.0,
                label="Hold Decisions",
            )
        if blocked_count_history:
            ax3.plot(
                x_counts[: len(blocked_count_history)],
                blocked_count_history,
                color=BLOCKED_COLOR,
                linewidth=2.0,
                linestyle="--",
                marker="o",
                markersize=3.0,
                label="Blocked by Constraints",
            )
        ax3.legend(frameon=False, loc="upper left")
    else:
        ax3.text(
            0.5, 0.5, "No action-count data", ha="center", va="center", fontsize=12
        )
    ax3.set_title("Agent Decisions by Step", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Count")

    ax4 = axes[1, 1]
    _style_axis(ax4)
    if wealth_pairs:
        agent_ids = [item[0] for item in wealth_pairs]
        wealth_values = [item[1] for item in wealth_pairs]
        wealth_colors = _blend_hex_colors(
            WEALTH_GRADIENT_START,
            WEALTH_GRADIENT_END,
            len(wealth_values),
        )
        ax4.bar(
            agent_ids,
            wealth_values,
            color=wealth_colors,
            edgecolor="#0f172a",
            linewidth=0.35,
            alpha=0.92,
        )
        average_wealth = sum(wealth_values) / len(wealth_values)
        ax4.axhline(
            average_wealth,
            color="#475569",
            linewidth=1.4,
            linestyle="--",
            label="Average Wealth",
        )
        _set_dynamic_ylim(ax4, wealth_values, pad_ratio=0.18, min_pad=8.0)
        ax4.tick_params(axis="x", rotation=45)
        ax4.legend(frameon=False, loc="upper left")
    else:
        ax4.text(
            0.5, 0.5, "No agent wealth data", ha="center", va="center", fontsize=12
        )
    ax4.set_title("Final Agent Wealth", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Agent")
    ax4.set_ylabel("Wealth")

    if price_history:
        initial_price = price_history[0]
        final_return = (
            (final_price - initial_price) / initial_price if initial_price != 0 else 0.0
        )
    else:
        final_return = 0.0

    footer_text = (
        f"Final Price={final_price:.4f} | "
        f"Final Return={final_return:.2%} | "
        f"Warm-up={summary.get('bootstrap_steps', 0)} | "
        f"Experiment={experiment_label}"
    )
    fig.text(0.5, 0.012, footer_text, ha="center", fontsize=10.5, color=TEXT_COLOR)

    figure_path = output_dir / "report.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight", facecolor=FIGURE_BG)
    plt.close(fig)

    return output_dir
