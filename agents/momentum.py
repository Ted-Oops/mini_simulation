from agents.base import BaseAgent
from models import (
    AgentObservation,
    DecisionMode,
    OrderDecision,
    StrategyType,
    TradeAction,
)


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
        self.entry_threshold = self._stable_uniform("entry_threshold", 0.010, 0.018)
        self.add_threshold = self._stable_uniform("add_threshold", 0.016, 0.028)
        self.exit_threshold = self._stable_uniform("exit_threshold", 0.006, 0.015)
        self.max_inventory_ratio = self._stable_uniform(
            "max_inventory_ratio", 0.18, 0.32
        )
        self.stop_loss_threshold = self._stable_uniform(
            "stop_loss_threshold", 0.018, 0.035
        )
        self.take_profit_threshold = self._stable_uniform(
            "take_profit_threshold", 0.025, 0.055
        )

    def _decide_rule_based(self, observation: AgentObservation) -> OrderDecision:
        trend_score = 0.65 * observation.momentum_1 + 0.35 * observation.momentum_3
        continuation_score = trend_score - 0.35 * max(-observation.reversal_score, 0.0)
        inventory_ratio = self.inventory_ratio(observation.price)
        unrealized_return = self.unrealized_return(observation.price)

        if (
            self.state.shares > 0
            and observation.volatility > 0.08
            and trend_score < 0.0
            and self.can_sell(1)
        ):
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.SELL,
                quantity=1,
                limit_price=observation.price,
                reason="trend has softened while volatility is elevated, so momentum exposure is reduced",
                signal_strength=min(observation.volatility / 0.08, 1.0),
            )

        if (
            self.state.shares > 0
            and (
                trend_score < -self.exit_threshold
                or observation.reversal_score < -0.025
                or unrealized_return < -self.stop_loss_threshold
            )
        ):
            if self.can_sell(1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.SELL,
                    quantity=1,
                    limit_price=observation.price,
                    reason="trend continuation has broken down, so the momentum position is trimmed",
                    signal_strength=min(
                        max(
                            abs(trend_score) / self.exit_threshold,
                            abs(unrealized_return) / self.stop_loss_threshold,
                        ),
                        1.0,
                    ),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="trend has weakened, but there is no remaining inventory to sell",
                signal_strength=min(abs(trend_score) / self.exit_threshold, 1.0),
            )

        if (
            self.state.shares > 0
            and unrealized_return > self.take_profit_threshold
            and observation.reversal_score < -0.010
            and self.can_sell(1)
        ):
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.SELL,
                quantity=1,
                limit_price=observation.price,
                reason="the trend has produced gains, but a near-term reversal signal suggests taking some profit",
                signal_strength=min(
                    unrealized_return / self.take_profit_threshold,
                    1.0,
                ),
            )

        if continuation_score > self.add_threshold:
            if (
                inventory_ratio < self.max_inventory_ratio
                and self.can_buy(observation.price, 1)
            ):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.BUY,
                    quantity=1,
                    limit_price=observation.price,
                    reason="trend continuation remains strong enough to add to the position",
                    signal_strength=min(continuation_score / self.add_threshold, 1.0),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="trend is positive, but inventory or cash limits prevent adding more exposure",
                signal_strength=min(continuation_score / self.add_threshold, 1.0),
            )

        if (
            inventory_ratio < 0.10
            and continuation_score > self.entry_threshold
            and self.can_buy(observation.price, 1)
        ):
            if self.can_buy(observation.price, 1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.BUY,
                    quantity=1,
                    limit_price=observation.price,
                    reason="trend continuation has just turned favorable enough to open a momentum position",
                    signal_strength=min(
                        continuation_score / self.entry_threshold,
                        1.0,
                    ),
                )

        return OrderDecision(
            agent_id=self.agent_id,
            action=TradeAction.HOLD,
            quantity=0,
            reason="the current trend is not strong enough to justify adding or reducing momentum exposure",
            signal_strength=min(
                abs(continuation_score) / max(self.entry_threshold, 1e-6),
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
