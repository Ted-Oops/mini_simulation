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

    def _decide_rule_based(self, observation: AgentObservation) -> OrderDecision:
        crowd_signal = observation.peer_buy_ratio - observation.peer_sell_ratio
        expected_return = (
            0.8 * crowd_signal
            + 0.02 * observation.peer_net_demand
            + 0.2 * observation.momentum_1
            - 0.1 * observation.reversal_score
            + 0.1 * observation.shock
        )

        if observation.volatility > 0.15:
            expected_return *= 0.5

        if expected_return > 0.01:
            if self.can_buy(observation.price, 1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.BUY,
                    quantity=1,
                    limit_price=observation.price,
                    reason="previous peer behavior implies short-term buying pressure",
                    signal_strength=abs(expected_return),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="positive speculative signal exists but cash is insufficient",
                signal_strength=abs(expected_return),
            )

        if expected_return < -0.01:
            if self.can_sell(1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.SELL,
                    quantity=1,
                    limit_price=observation.price,
                    reason="previous peer behavior implies short-term selling pressure",
                    signal_strength=abs(expected_return),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="negative speculative signal exists but no shares are available",
                signal_strength=abs(expected_return),
            )

        return OrderDecision(
            agent_id=self.agent_id,
            action=TradeAction.HOLD,
            quantity=0,
            reason="expected short-term return is too small",
            signal_strength=abs(expected_return),
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
