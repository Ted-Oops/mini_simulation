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

    def _decide_rule_based(self, observation: AgentObservation) -> OrderDecision:
        if observation.sue_signal > 0.20:
            if self.can_buy(observation.price, 1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.BUY,
                    quantity=1,
                    limit_price=observation.price,
                    reason="simulated SUE signal is sufficiently positive",
                    signal_strength=abs(observation.sue_signal),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="buy signal exists but cash is insufficient",
                signal_strength=abs(observation.sue_signal),
            )

        if observation.sue_signal < -0.20:
            if self.can_sell(1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.SELL,
                    quantity=1,
                    limit_price=observation.price,
                    reason="simulated SUE signal is sufficiently negative",
                    signal_strength=abs(observation.sue_signal),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="sell signal exists but no shares are available",
                signal_strength=abs(observation.sue_signal),
            )

        return OrderDecision(
            agent_id=self.agent_id,
            action=TradeAction.HOLD,
            quantity=0,
            reason="simulated SUE signal is not strong enough",
            signal_strength=abs(observation.sue_signal),
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
