from models import AgentObservation, AgentState, DecisionMode, OrderDecision, StrategyType, TradeAction


class BaseAgent:
    def __init__(
            self,
            agent_id: str,
            strategy_type: StrategyType,
            decision_mode: DecisionMode,
            initial_cash: float
    ) -> None:
        self.agent_id = agent_id
        self.strategy_type = strategy_type
        self.decision_mode = decision_mode
        self.state = AgentState(
            agent_id=agent_id,
            cash=initial_cash,
            shares=0,
            avg_cost=0.0
        )

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
            return None

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
            return None

        if decision.action == TradeAction.SELL:
            if not self.can_sell(quantity):
                raise ValueError(
                    f"{self.agent_id} does not have enough shares to sell {quantity}"
                )

            self.state.cash += trade_value
            self.state.shares -= quantity

            if self.state.shares == 0:
                self.state.avg_cost = 0.0
            return None

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


class FundamentalAgent(BaseAgent):
    def __init__(
            self,
            agent_id: str,
            decision_mode: DecisionMode,
            initial_cash: float
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            strategy_type=StrategyType.FUNDAMENTAL,
            decision_mode=decision_mode,
            initial_cash=initial_cash
        )

    def _decide_rule_based(self, observation: AgentObservation) -> OrderDecision:
        if observation.mispricing < -0.05:
            if self.can_buy(observation.price, 1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.BUY,
                    quantity=1,
                    limit_price=observation.price,
                    reason="price is sufficiently below fundamental value",
                    signal_strength=abs(observation.mispricing),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="buy signal exists but cash is insufficient",
                signal_strength=abs(observation.mispricing),
            )

        if observation.mispricing > 0.05:
            if self.can_sell(1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.SELL,
                    quantity=1,
                    limit_price=observation.price,
                    reason="price is sufficiently above fundamental value",
                    signal_strength=abs(observation.mispricing),
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="sell signal exists but no shares are available",
                signal_strength=abs(observation.mispricing),
            )

        return OrderDecision(
            agent_id=self.agent_id,
            action=TradeAction.HOLD,
            quantity=0,
            reason="mispricing is not strong enough",
            signal_strength=abs(observation.mispricing),
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


class MomentumAgent(BaseAgent):
    def __init__(
            self,
            agent_id: str,
            decision_mode: DecisionMode,
            initial_cash: float
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            strategy_type=StrategyType.MOMENTUM,
            decision_mode=decision_mode,
            initial_cash=initial_cash
        )

    def _decide_rule_based(self, observation: AgentObservation) -> OrderDecision:
        bullish_trend = observation.momentum_1 > 0 and observation.momentum_3 > 0
        bearish_trend = observation.momentum_1 < 0 and observation.momentum_3 < 0
        trend_strength = abs(observation.momentum_1) + abs(observation.momentum_3)

        if observation.volatility > 0.10:
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="volatility is too high for the current momentum template",
                signal_strength=observation.volatility,
            )

        if bullish_trend:
            if self.can_buy(observation.price, 1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.BUY,
                    quantity=1,
                    limit_price=observation.price,
                    reason="short-term and medium-term momentum are both positive",
                    signal_strength=trend_strength,
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="bullish trend exists but cash is insufficient",
                signal_strength=trend_strength,
            )

        if bearish_trend:
            if self.can_sell(1):
                return OrderDecision(
                    agent_id=self.agent_id,
                    action=TradeAction.SELL,
                    quantity=1,
                    limit_price=observation.price,
                    reason="short-term and medium-term momentum are both negative",
                    signal_strength=trend_strength,
                )
            return OrderDecision(
                agent_id=self.agent_id,
                action=TradeAction.HOLD,
                quantity=0,
                reason="bearish trend exists but no shares are available",
                signal_strength=trend_strength,
            )

        return OrderDecision(
            agent_id=self.agent_id,
            action=TradeAction.HOLD,
            quantity=0,
            reason="momentum signals are mixed",
            signal_strength=trend_strength,
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


class SpeculatorAgent(BaseAgent):
    def __init__(
            self,
            agent_id: str,
            decision_mode: DecisionMode,
            initial_cash: float
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            strategy_type=StrategyType.SPECULATOR,
            decision_mode=decision_mode,
            initial_cash=initial_cash
        )

    def _decide_rule_based(self, observation: AgentObservation) -> OrderDecision:
        expected_return = (
            0.7 * observation.momentum_1
            + 0.3 * observation.momentum_3
            + 0.001 * observation.net_demand
            - 0.4 * observation.mispricing
            + 0.2 * observation.shock
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
                    reason="expected short-term return is positive in the current template",
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
                    reason="expected short-term return is negative in the current template",
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
