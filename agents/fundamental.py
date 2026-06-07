import math

from agents.base import BaseAgent
from models import (
    AgentObservation,
    DecisionMode,
    OrderDecision,
    StrategyType,
    TradeAction,
)


class FundamentalAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        decision_mode: DecisionMode,
        initial_cash: float,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            strategy_type=StrategyType.FUNDAMENTAL,
            decision_mode=decision_mode,
            initial_cash=initial_cash,
        )
        self.value_sensitivity = self._stable_uniform(
            "value_sensitivity", 0.045, 0.075
        )
        self.entry_threshold = self._stable_uniform("entry_threshold", 0.010, 0.018)
        self.exit_threshold = self._stable_uniform("exit_threshold", 0.010, 0.020)
        self.max_inventory_ratio = self._stable_uniform(
            "max_inventory_ratio", 0.45, 0.65
        )
        self.profit_take_threshold = self._stable_uniform(
            "profit_take_threshold", 0.030, 0.060
        )

    def _decide_rule_based(self, observation: AgentObservation) -> OrderDecision:
        fair_value = observation.price * (
            1 + self.value_sensitivity * math.tanh(observation.sue_signal)
        )
        valuation_gap = (
            (fair_value - observation.price) / observation.price
            if observation.price > 0
            else 0.0
        )
        inventory_ratio = self.inventory_ratio(observation.price)
        unrealized_return = self.unrealized_return(observation.price)

        if valuation_gap > self.entry_threshold:
            if (
                inventory_ratio < self.max_inventory_ratio
                and self.can_buy(observation.price, 1)
            ):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.BUY,
                    quantity=1,
                    limit_price=observation.price,
                    reason="estimated fair value is sufficiently above the current market price",
                    signal_strength=min(abs(valuation_gap) / self.entry_threshold, 1.0),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="the asset looks undervalued, but current inventory or cash constraints block a new purchase",
                signal_strength=min(abs(valuation_gap) / self.entry_threshold, 1.0),
            )

        if valuation_gap < -self.exit_threshold:
            if self.can_sell(1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.SELL,
                    quantity=1,
                    limit_price=observation.price,
                    reason="estimated fair value is sufficiently below the current market price",
                    signal_strength=min(abs(valuation_gap) / self.exit_threshold, 1.0),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="the asset looks overvalued, but no shares are available to reduce exposure",
                signal_strength=min(abs(valuation_gap) / self.exit_threshold, 1.0),
            )

        if (
            self.state.shares > 0
            and unrealized_return > self.profit_take_threshold
            and (observation.reversal_score < -0.01 or observation.momentum_1 < 0.0)
        ):
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.SELL,
                quantity=1,
                limit_price=observation.price,
                reason="price has moved materially above cost while short-term conditions suggest mean reversion risk",
                signal_strength=min(unrealized_return / self.profit_take_threshold, 1.0),
            )

        return OrderDecision(
            agent_id=self.agent_id,
            action=TradeAction.HOLD,
            quantity=0,
            reason="price is close to estimated fair value, so no rebalance is needed",
            signal_strength=min(abs(valuation_gap) / max(self.entry_threshold, 1e-6), 1.0),
        )

    def _decide_half_rule_based(self, observation: AgentObservation) -> OrderDecision:
        rule_decision = self._decide_rule_based(observation)
        return self._decide_with_llm(
            observation=observation,
            decision_mode=DecisionMode.HALF_RULE_BASED,
            allowed_actions=self._half_rule_allowed_actions(
                rule_decision,
                observation,
            ),
            fallback_decision=rule_decision,
            rule_decision=rule_decision,
        )

    def _decide_open_ended(self, observation: AgentObservation) -> OrderDecision:
        fallback_decision = self._decide_rule_based(observation)
        return self._decide_with_llm(
            observation=observation,
            decision_mode=DecisionMode.OPEN_ENDED,
            allowed_actions=self._legal_trade_actions(observation),
            fallback_decision=fallback_decision,
            rule_decision=None,
        )
