from dataclasses import dataclass

from agents import BaseAgent
from market.order_book import OrderBook
from models import ExperimentConfig, LimitOrder, MarketState, TradeExecution


@dataclass(frozen=True)
class MarketHistories:
    buy_count_history: list[int]
    sell_count_history: list[int]
    hold_count_history: list[int]
    rejected_count_history: list[int]
    trade_count_history: list[int]
    batch_count_history: list[int]
    expired_order_count_history: list[int]
    open_order_count_history: list[int]
    best_bid_history: list[float | None]
    best_ask_history: list[float | None]
    avg_wealth_history: list[float]
    sue_history: list[float]
    reversal_history: list[float]
    llm_success_count_history: list[int]
    llm_fallback_count_history: list[int]


def serialize_order(order: LimitOrder) -> dict:
    """把订单对象转成可写入摘要文件的字典。"""

    return {
        "order_id": order.order_id,
        "agent_id": order.agent_id,
        "action": order.action.value,
        "quantity": order.quantity,
        "limit_price": order.limit_price,
        "submitted_step": order.submitted_step,
        "submitted_batch": order.submitted_batch,
        "sequence": order.sequence,
        "reason": order.reason,
        "signal_strength": order.signal_strength,
    }


def serialize_trade(trade: TradeExecution) -> dict:
    """把成交对象转成可写入摘要文件的字典。"""

    return {
        "trade_id": trade.trade_id,
        "buy_order_id": trade.buy_order_id,
        "sell_order_id": trade.sell_order_id,
        "price": trade.price,
        "quantity": trade.quantity,
        "step": trade.step,
        "batch": trade.batch,
        "sequence": trade.sequence,
        "buyer_id": trade.buyer_id,
        "seller_id": trade.seller_id,
    }


def build_summary(
    *,
    config: ExperimentConfig,
    bootstrap_steps: int,
    market_state: MarketState,
    agents: list[BaseAgent],
    order_book: OrderBook,
    histories: MarketHistories,
    llm_fallback_reasons: dict[str, int],
) -> dict:
    """导出实验摘要。"""

    simulation_price_history = market_state.price_history[bootstrap_steps:]
    simulation_volume_history = market_state.volume_history[bootstrap_steps:]
    simulation_net_demand_history = market_state.net_demand_history[bootstrap_steps:]
    llm_success_count = sum(histories.llm_success_count_history)
    llm_fallback_count = sum(histories.llm_fallback_count_history)
    llm_decision_count = llm_success_count + llm_fallback_count

    return {
        "experiment_label": f"{config.strategy_type.value}__{config.decision_mode.value}",
        "strategy_type": config.strategy_type.value,
        "decision_mode": config.decision_mode.value,
        "num_agents": len(agents),
        "num_steps": market_state.step,
        "bootstrap_steps": bootstrap_steps,
        "initial_shares_per_agent": config.initial_shares_per_agent,
        "batch_size": config.batch_size,
        "max_order_age_steps": config.max_order_age_steps,
        "price_tick": config.price_tick,
        "llm_stats": {
            "decision_count": llm_decision_count,
            "success_count": llm_success_count,
            "fallback_count": llm_fallback_count,
            "success_rate": llm_success_count / llm_decision_count if llm_decision_count else None,
            "fallback_reasons": llm_fallback_reasons,
        },
        "llm_success_count_history": list(histories.llm_success_count_history),
        "llm_fallback_count_history": list(histories.llm_fallback_count_history),
        "final_price": market_state.price,
        "final_mark_price": market_state.mark_price,
        "price_history": simulation_price_history,
        "submitted_net_demand_history": simulation_net_demand_history,
        "mark_price_history": market_state.mark_price_history[bootstrap_steps:],
        "volume_history": simulation_volume_history,
        "net_demand_history": simulation_net_demand_history,
        "bootstrap_price_history": market_state.price_history[: bootstrap_steps + 1],
        "bootstrap_mark_price_history": market_state.mark_price_history[
            : bootstrap_steps + 1
        ],
        "bootstrap_return_history": market_state.return_history[:bootstrap_steps],
        "bootstrap_volume_history": market_state.volume_history[:bootstrap_steps],
        "bootstrap_net_demand_history": market_state.net_demand_history[
            :bootstrap_steps
        ],
        "buy_count_history": histories.buy_count_history,
        "sell_count_history": histories.sell_count_history,
        "hold_count_history": histories.hold_count_history,
        "rejected_count_history": histories.rejected_count_history,
        "trade_count_history": histories.trade_count_history,
        "batch_count_history": histories.batch_count_history,
        "expired_order_count_history": histories.expired_order_count_history,
        "open_order_count_history": histories.open_order_count_history,
        "best_bid_history": histories.best_bid_history,
        "best_ask_history": histories.best_ask_history,
        "avg_wealth_history": histories.avg_wealth_history,
        "sue_history": histories.sue_history,
        "reversal_history": histories.reversal_history,
        "trade_history": [
            serialize_trade(trade) for trade in order_book.trades
        ],
        "open_buy_orders": [
            serialize_order(order) for order in order_book.buy_orders
        ],
        "open_sell_orders": [
            serialize_order(order) for order in order_book.sell_orders
        ],
        "agents": [
            {
                "agent_id": agent.agent_id,
                "cash": agent.state.cash,
                "shares": agent.state.shares,
                "avg_cost": agent.state.avg_cost,
                "wealth": agent.state.cash + agent.state.shares * market_state.price,
                "mark_wealth": agent.state.cash
                + agent.state.shares * market_state.mark_price,
            }
            for agent in agents
        ],
    }
