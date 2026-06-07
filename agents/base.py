import hashlib
from typing import Any

from llm_client import get_default_llm_client
from models import (
    AgentObservation,
    AgentState,
    DecisionMode,
    OrderDecision,
    StrategyType,
    TradeAction,
)
from prompts import build_llm_decision_messages


class BaseAgent:
    LLM_LIMIT_PRICE_OFFSET_RATIO = 0.04

    def __init__(
        self,
        agent_id: str,
        strategy_type: StrategyType,
        decision_mode: DecisionMode,
        initial_cash: float,
    ) -> None:
        self.agent_id = agent_id
        self.strategy_type = strategy_type
        self.decision_mode = decision_mode
        self.state = AgentState(
            agent_id=agent_id,
            cash=initial_cash,
            shares=0,
            avg_cost=0.0,
        )

    def _stable_fraction(self, key: str) -> float:
        payload = f"{self.strategy_type.value}:{self.agent_id}:{key}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return integer / ((1 << 64) - 1)

    def _stable_uniform(self, key: str, low: float, high: float) -> float:
        return low + (high - low) * self._stable_fraction(key)

    def wealth(self, price: float) -> float:
        return self.state.cash + self.state.shares * max(price, 0.0)

    def inventory_ratio(self, price: float) -> float:
        total_wealth = self.wealth(price)
        if total_wealth <= 0:
            return 0.0
        return (self.state.shares * max(price, 0.0)) / total_wealth

    def unrealized_return(self, price: float) -> float:
        if self.state.shares <= 0 or self.state.avg_cost <= 0:
            return 0.0
        return (price - self.state.avg_cost) / self.state.avg_cost

    def can_buy(self, price: float, quantity: int) -> bool:
        if price <= 0 or quantity <= 0:
            return False
        return self.state.cash >= price * quantity

    def can_sell(self, quantity: int) -> bool:
        if quantity <= 0:
            return False
        return self.state.shares >= quantity

    def apply_trade(self, decision: OrderDecision, execution_price: float) -> None:
        if decision.action == TradeAction.HOLD:
            return

        if execution_price <= 0:
            raise ValueError(f"Execution price must be positive, got {execution_price}")

        quantity = decision.quantity
        trade_value = execution_price * quantity

        if decision.action == TradeAction.BUY:
            if not self.can_buy(execution_price, quantity):
                raise ValueError(
                    f"{self.agent_id} does not have enough cash to buy {quantity} shares"
                )

            old_shares = self.state.shares
            old_avg_cost = self.state.avg_cost

            self.state.cash -= trade_value
            self.state.shares += quantity

            if old_shares == 0:
                self.state.avg_cost = execution_price
            else:
                total_cost = old_shares * old_avg_cost + quantity * execution_price
                self.state.avg_cost = total_cost / self.state.shares
            return

        if decision.action == TradeAction.SELL:
            if not self.can_sell(quantity):
                raise ValueError(
                    f"{self.agent_id} does not have enough shares to sell {quantity}"
                )

            self.state.cash += trade_value
            self.state.shares -= quantity

            if self.state.shares == 0:
                self.state.avg_cost = 0.0
            return

        raise ValueError(f"Unsupported trade action: {decision.action}")

    def decide(self, observation: AgentObservation) -> OrderDecision:
        if self.decision_mode == DecisionMode.RULE_BASED:
            return self._decide_rule_based(observation)

        if self.decision_mode == DecisionMode.HALF_RULE_BASED:
            return self._decide_half_rule_based(observation)

        if self.decision_mode == DecisionMode.OPEN_ENDED:
            return self._decide_open_ended(observation)

        raise ValueError(f"Unsupported decision mode: {self.decision_mode}")

    def _decide_rule_based(self, observation: AgentObservation) -> OrderDecision:
        raise NotImplementedError

    def _decide_half_rule_based(self, observation: AgentObservation) -> OrderDecision:
        raise NotImplementedError

    def _decide_open_ended(self, observation: AgentObservation) -> OrderDecision:
        raise NotImplementedError

    def _decide_with_llm(
        self,
        *,
        observation: AgentObservation,
        decision_mode: DecisionMode,
        allowed_actions: list[str],
        fallback_decision: OrderDecision,
        rule_decision: OrderDecision | None = None,
    ) -> OrderDecision:
        max_quantity = self._max_trade_quantity(observation)
        messages = build_llm_decision_messages(
            strategy_type=self.strategy_type,
            decision_mode=decision_mode,
            agent_snapshot=self._agent_snapshot(observation),
            observation=observation,
            allowed_actions=allowed_actions,
            max_quantity=max_quantity,
            rule_decision=rule_decision,
        )
        client = get_default_llm_client()
        payload = client.complete_json(messages)
        if payload is None:
            reason = client.last_error or "LLM unavailable"
            return self._fallback_decision(
                fallback_decision,
                f"{decision_mode.value} fallback because {reason}",
            )

        return self._validated_llm_decision(
            payload=payload,
            observation=observation,
            allowed_actions=allowed_actions,
            fallback_decision=fallback_decision,
            decision_mode=decision_mode,
        )

    def _legal_trade_actions(self, observation: AgentObservation) -> list[str]:
        actions = [TradeAction.HOLD.value]
        if self.can_buy(observation.price, 1):
            actions.append(TradeAction.BUY.value)
        if self.can_sell(1):
            actions.append(TradeAction.SELL.value)
        return actions

    def _half_rule_allowed_actions(
        self,
        rule_decision: OrderDecision,
        observation: AgentObservation,
    ) -> list[str]:
        legal_actions = self._legal_trade_actions(observation)
        if rule_decision.action != TradeAction.HOLD:
            return [
                action
                for action in [TradeAction.HOLD.value, rule_decision.action.value]
                if action in legal_actions
            ]
        return legal_actions

    def _agent_snapshot(self, observation: AgentObservation) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "cash": round(self.state.cash, 4),
            "shares": self.state.shares,
            "avg_cost": round(self.state.avg_cost, 4),
            "wealth": round(self.wealth(observation.price), 4),
            "inventory_ratio": round(self.inventory_ratio(observation.price), 4),
            "unrealized_return": round(self.unrealized_return(observation.price), 4),
        }

    def _max_trade_quantity(self, observation: AgentObservation) -> int:
        max_buy_quantity = (
            int(self.state.cash // observation.price) if observation.price > 0 else 0
        )
        return max(0, min(1, max(max_buy_quantity, self.state.shares)))

    def _fallback_decision(
        self,
        fallback_decision: OrderDecision,
        reason_prefix: str,
    ) -> OrderDecision:
        reason = f"{reason_prefix}; local decision: {fallback_decision.reason}"
        return OrderDecision(
            agent_id=self.agent_id,
            action=fallback_decision.action,
            quantity=fallback_decision.quantity,
            limit_price=fallback_decision.limit_price,
            reason=reason[:260],
            signal_strength=fallback_decision.signal_strength,
        )

    def _validated_llm_decision(
        self,
        *,
        payload: dict[str, Any],
        observation: AgentObservation,
        allowed_actions: list[str],
        fallback_decision: OrderDecision,
        decision_mode: DecisionMode,
    ) -> OrderDecision:
        action_text = str(payload.get("action", "")).strip().lower()
        if action_text not in allowed_actions:
            return self._fallback_decision(
                fallback_decision,
                f"{decision_mode.value} fallback because LLM action was not allowed",
            )

        action = TradeAction(action_text)
        if action == TradeAction.HOLD:
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                limit_price=None,
                reason=self._llm_reason(payload, decision_mode),
                signal_strength=self._llm_signal_strength(payload, fallback_decision),
            )

        max_quantity = self._max_quantity_for_action(action, observation.price)
        requested_quantity = self._coerce_positive_int(payload.get("quantity"), 1)
        quantity = min(requested_quantity, max_quantity)
        if quantity <= 0:
            return self._fallback_decision(
                fallback_decision,
                f"{decision_mode.value} fallback because LLM quantity was infeasible",
            )

        limit_price = self._clamped_limit_price(
            payload.get("limit_price"),
            observation.price,
            action,
            quantity,
        )
        if limit_price is None:
            return self._fallback_decision(
                fallback_decision,
                f"{decision_mode.value} fallback because LLM limit price was invalid",
            )

        return OrderDecision(
            agent_id=self.agent_id,
            action=action,
            quantity=quantity,
            limit_price=limit_price,
            reason=self._llm_reason(payload, decision_mode),
            signal_strength=self._llm_signal_strength(payload, fallback_decision),
        )

    def _max_quantity_for_action(self, action: TradeAction, price: float) -> int:
        if action == TradeAction.BUY:
            if price <= 0:
                return 0
            return max(0, min(1, int(self.state.cash // price)))
        if action == TradeAction.SELL:
            return max(0, min(1, self.state.shares))
        return 0

    def _clamped_limit_price(
        self,
        raw_limit_price: Any,
        reference_price: float,
        action: TradeAction,
        quantity: int,
    ) -> float | None:
        if reference_price <= 0:
            return None

        limit_price = self._coerce_float(raw_limit_price, reference_price)
        if limit_price <= 0:
            limit_price = reference_price

        lower_bound = reference_price * (1 - self.LLM_LIMIT_PRICE_OFFSET_RATIO)
        upper_bound = reference_price * (1 + self.LLM_LIMIT_PRICE_OFFSET_RATIO)
        limit_price = min(max(limit_price, lower_bound), upper_bound)

        if action == TradeAction.BUY:
            max_affordable_price = self.state.cash / quantity if quantity > 0 else 0.0
            limit_price = min(limit_price, max_affordable_price)

        return limit_price if limit_price > 0 else None

    @staticmethod
    def _coerce_positive_int(value: Any, default: int) -> int:
        try:
            return max(0, int(round(float(value))))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _llm_signal_strength(
        payload: dict[str, Any],
        fallback_decision: OrderDecision,
    ) -> float:
        raw_value = payload.get(
            "signal_strength",
            payload.get("confidence", fallback_decision.signal_strength),
        )
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = fallback_decision.signal_strength
        return max(0.0, min(value, 1.0))

    @staticmethod
    def _llm_reason(payload: dict[str, Any], decision_mode: DecisionMode) -> str:
        reason = str(payload.get("reason") or "LLM decision")
        reason = " ".join(reason.split())
        return f"{decision_mode.value} LLM: {reason}"[:260]
