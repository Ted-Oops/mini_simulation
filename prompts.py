from __future__ import annotations

import json
from typing import Any

from models import (
    AgentObservation,
    DecisionMode,
    OrderDecision,
    StrategyType,
)


STRATEGY_GUIDES = {
    StrategyType.FUNDAMENTAL: (
        "Trade around estimated fair value. Positive sue_signal suggests upside "
        "fundamental news; negative sue_signal suggests downside risk. Prefer "
        "gradual rebalancing over aggressive inventory swings."
    ),
    StrategyType.MOMENTUM: (
        "Trade trend continuation while respecting stop-loss and take-profit "
        "discipline. Positive momentum supports buying; negative momentum or "
        "reversal risk supports reducing exposure."
    ),
    StrategyType.SPECULATOR: (
        "Trade short-term order-flow opportunities. Peer buy pressure and "
        "positive near-term price action support buying; peer sell pressure, "
        "negative momentum, or reversal risk support quick inventory reduction."
    ),
}


def _decision_payload(decision: OrderDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "action": decision.action.value,
        "quantity": decision.quantity,
        "limit_price": decision.limit_price,
        "reason": decision.reason,
        "signal_strength": round(decision.signal_strength, 4),
    }


def _observation_payload(observation: AgentObservation) -> dict[str, Any]:
    return {
        "step": observation.step,
        "price": round(observation.price, 4),
        "sue_signal": round(observation.sue_signal, 4),
        "momentum_1": round(observation.momentum_1, 4),
        "momentum_3": round(observation.momentum_3, 4),
        "reversal_score": round(observation.reversal_score, 4),
        "volatility": round(observation.volatility, 4),
        "net_demand": round(observation.net_demand, 4),
        "peer_buy_ratio": round(observation.peer_buy_ratio, 4),
        "peer_sell_ratio": round(observation.peer_sell_ratio, 4),
        "peer_hold_ratio": round(observation.peer_hold_ratio, 4),
        "peer_net_demand": round(observation.peer_net_demand, 4),
        "shock": round(observation.shock, 4),
    }


def build_llm_decision_messages(
    *,
    strategy_type: StrategyType,
    decision_mode: DecisionMode,
    agent_snapshot: dict[str, Any],
    observation: AgentObservation,
    allowed_actions: list[str],
    max_quantity: int,
    rule_decision: OrderDecision | None = None,
) -> list[dict[str, str]]:
    if decision_mode == DecisionMode.HALF_RULE_BASED:
        mode_instruction = (
            "You are in half_rule_based mode. The local rule engine provides a "
            "baseline decision. You may accept it, choose HOLD, or make a small "
            "legal adjustment only when the market context clearly justifies it."
        )
    else:
        mode_instruction = (
            "You are in open_ended mode. You may choose any legal action, but "
            "the decision must remain conservative and consistent with the "
            "agent's trading style and inventory constraints."
        )

    context = {
        "strategy_type": strategy_type.value,
        "decision_mode": decision_mode.value,
        "strategy_guide": STRATEGY_GUIDES[strategy_type],
        "agent_state": agent_snapshot,
        "observation": _observation_payload(observation),
        "rule_baseline_decision": _decision_payload(rule_decision),
        "constraints": {
            "allowed_actions": allowed_actions,
            "max_quantity": max_quantity,
            "quantity_must_be_zero_when_holding": True,
            "limit_price_can_be_null": True,
        },
    }

    system_content = (
        "You are a trading decision module in a toy limit-order-book market. "
        "Return exactly one JSON object and no markdown. Required schema: "
        '{"action":"buy|sell|hold","quantity":0,"limit_price":null,'
        '"reason":"brief reason","signal_strength":0.0}. '
        "Use only the allowed actions. signal_strength must be between 0 and 1."
    )
    user_content = (
        f"{mode_instruction}\n\n"
        "Market context JSON:\n"
        f"{json.dumps(context, ensure_ascii=True, indent=2)}"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
