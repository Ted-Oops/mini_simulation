from agents.base import BaseAgent
from models import AgentObservation, DecisionMode, OrderDecision, StrategyType, TradeAction


class MomentumAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        decision_mode: DecisionMode,
        initial_cash: float,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            strategy_type=StrategyType.MOMENTUM,
            decision_mode=decision_mode,
            initial_cash=initial_cash,
        )

    def _decide_rule_based(self, observation: AgentObservation) -> OrderDecision:
        trend_score = 0.7 * observation.momentum_1 + 0.3 * observation.momentum_3
        technical_score = trend_score + 0.5 * observation.reversal_score

        if observation.volatility > 0.10:
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="volatility is too high for the current momentum template",
                signal_strength=observation.volatility,
            )

        if observation.reversal_score > 0.03:
            if self.can_buy(observation.price, 1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.BUY,
                    quantity=1,
                    limit_price=observation.price,
                    reason="reversal indicator suggests rebound potential",
                    signal_strength=abs(observation.reversal_score),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="bullish reversal exists but cash is insufficient",
                signal_strength=abs(observation.reversal_score),
            )

        if observation.reversal_score < -0.03:
            if self.can_sell(1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.SELL,
                    quantity=1,
                    limit_price=observation.price,
                    reason="reversal indicator suggests pullback risk",
                    signal_strength=abs(observation.reversal_score),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="bearish reversal exists but no shares are available",
                signal_strength=abs(observation.reversal_score),
            )

        if technical_score > 0.01:
            if self.can_buy(observation.price, 1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.BUY,
                    quantity=1,
                    limit_price=observation.price,
                    reason="trend and reversal filters jointly support a long position",
                    signal_strength=abs(technical_score),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="bullish technical score exists but cash is insufficient",
                signal_strength=abs(technical_score),
            )

        if technical_score < -0.01:
            if self.can_sell(1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.SELL,
                    quantity=1,
                    limit_price=observation.price,
                    reason="trend and reversal filters jointly support a short-term exit",
                    signal_strength=abs(technical_score),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="bearish technical score exists but no shares are available",
                signal_strength=abs(technical_score),
            )

        return OrderDecision(
            agent_id=self.agent_id,
            action=TradeAction.HOLD,
            quantity=0,
            reason="trend and reversal signals do not form a clear setup",
            signal_strength=abs(technical_score),
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
