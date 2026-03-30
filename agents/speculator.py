from agents.base import BaseAgent
from models import (
    AgentObservation,
    DecisionMode,
    OrderDecision,
    StrategyType,
    TradeAction,
)


class SpeculatorAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        decision_mode: DecisionMode,
        initial_cash: float,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            strategy_type=StrategyType.SPECULATOR,
            decision_mode=decision_mode,
            initial_cash=initial_cash,
        )
        self.crowd_weight = self._stable_uniform("crowd_weight", 0.45, 0.75)
        self.price_weight = self._stable_uniform("price_weight", 0.20, 0.45)
        self.entry_threshold = self._stable_uniform("entry_threshold", 0.040, 0.085)
        self.exit_threshold = self._stable_uniform("exit_threshold", 0.035, 0.075)
        self.max_inventory_ratio = self._stable_uniform(
            "max_inventory_ratio", 0.16, 0.30
        )
        self.take_profit_threshold = self._stable_uniform(
            "take_profit_threshold", 0.020, 0.050
        )

    def _decide_rule_based(self, observation: AgentObservation) -> OrderDecision:
        crowd_signal = observation.peer_buy_ratio - observation.peer_sell_ratio
        opportunity_score = (
            self.crowd_weight * crowd_signal
            + 0.02 * observation.peer_net_demand
            + self.price_weight * observation.momentum_1
            - 0.35 * observation.reversal_score
        )
        inventory_ratio = self.inventory_ratio(observation.price)
        unrealized_return = self.unrealized_return(observation.price)

        if observation.volatility > 0.09:
            opportunity_score *= 0.70

        if (
            self.state.shares > 0
            and (
                opportunity_score < -self.exit_threshold
                or (
                    unrealized_return > self.take_profit_threshold
                    and observation.reversal_score < -0.010
                )
            )
        ):
            if self.can_sell(1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.SELL,
                    quantity=1,
                    limit_price=observation.price,
                    reason="short-term speculative edge has turned negative, so inventory is reduced quickly",
                    signal_strength=min(
                        max(
                            abs(opportunity_score) / self.exit_threshold,
                            unrealized_return / max(self.take_profit_threshold, 1e-6),
                        ),
                        1.0,
                    ),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="the speculative edge is negative, but there is no inventory left to reduce",
                signal_strength=min(abs(opportunity_score) / self.exit_threshold, 1.0),
            )

        if opportunity_score > self.entry_threshold:
            if (
                inventory_ratio < self.max_inventory_ratio
                and self.can_buy(observation.price, 1)
            ):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.BUY,
                    quantity=1,
                    limit_price=observation.price,
                    reason="order-flow and price action jointly imply a positive short-term speculative edge",
                    signal_strength=min(
                        opportunity_score / self.entry_threshold,
                        1.0,
                    ),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="speculative edge is positive, but inventory or cash constraints block a new trade",
                signal_strength=min(
                    opportunity_score / self.entry_threshold,
                    1.0,
                ),
            )

        return OrderDecision(
            agent_id=self.agent_id,
            action=TradeAction.HOLD,
            quantity=0,
            reason="short-term order-flow and price signals do not offer enough speculative edge",
            signal_strength=min(
                abs(opportunity_score) / max(self.entry_threshold, 1e-6),
                1.0,
            ),
        )

    def _decide_half_rule_based(self, observation: AgentObservation) -> OrderDecision:
        return OrderDecision(
            agent_id=self.agent_id,
            action=TradeAction.HOLD,
            quantity=0,
            reason="half-rule-based mode is not implemented yet",
            signal_strength=0.0,
        )

    def _decide_open_ended(self, observation: AgentObservation) -> OrderDecision:
        return OrderDecision(
            agent_id=self.agent_id,
            action=TradeAction.HOLD,
            quantity=0,
            reason="open-ended mode is not implemented yet",
            signal_strength=0.0,
        )
