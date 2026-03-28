from models import (
    AgentObservation,
    AgentState,
    DecisionMode,
    OrderDecision,
    StrategyType,
    TradeAction,
)


class BaseAgent:
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
