import random

from agents import BaseAgent, FundamentalAgent, MomentumAgent, SpeculatorAgent
from models import ExperimentConfig, MarketState, StrategyType, AgentObservation, TradeAction


class SimpleMarket:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

        self.market_state = MarketState(
            step=0,
            price=config.initial_price,
            fundamental_value=config.initial_fundamental_value,
            price_history=[config.initial_price],
            return_history=[],
            volume_history=[],
            net_demand_history=[],
            shock=0.0
        )

        self.agents = self._create_agents()
        self.buy_count_history = []
        self.sell_count_history = []
        self.hold_count_history = []
        self.rejected_count_history = []
        self.avg_wealth_history = []
        self.mispricing_history = []
        self.fundamental_value_history = [config.initial_fundamental_value]


    def _create_agents(self) -> list[BaseAgent]:
        agents: list[BaseAgent] = []
        agent_prefix = self.config.strategy_type.value

        if self.config.strategy_type == StrategyType.FUNDAMENTAL:
            agent_cls = FundamentalAgent
        elif self.config.strategy_type == StrategyType.MOMENTUM:
            agent_cls = MomentumAgent
        elif self.config.strategy_type == StrategyType.SPECULATOR:
            agent_cls = SpeculatorAgent
        else:
            raise NotImplementedError(
                f"Strategy type {self.config.strategy_type} is not implemented yet"
            )

        for i in range(self.config.num_agents):
            agents.append(
                agent_cls(
                    agent_id=f"{agent_prefix}_{i + 1}",
                    decision_mode=self.config.decision_mode,
                    initial_cash=self.config.initial_cash
                )
            )
        return agents

    def _build_observation(self) -> AgentObservation:
        state = self.market_state

        if state.fundamental_value == 0:
            mispricing = 0.0
        else:
            mispricing = (state.price - state.fundamental_value) / state.fundamental_value

        if len(state.return_history) >= 1:
            momentum_1 = state.return_history[-1]
        else:
            momentum_1 = 0.0

        if len(state.return_history) >= 3:
            momentum_3 = sum(state.return_history[-3:]) / 3
        elif len(state.return_history) > 0:
            momentum_3 = sum(state.return_history) / len(state.return_history)
        else:
            momentum_3 = 0.0

        if len(state.return_history) >= 2:
            mean_return = sum(state.return_history) / len(state.return_history)
            squared_diffs = [
                (ret - mean_return) ** 2 for ret in state.return_history
            ]
            volatility = (sum(squared_diffs) / len(squared_diffs)) ** 0.5
        else:
            volatility = 0.0

        if len(state.net_demand_history) >= 1:
            net_demand = state.net_demand_history[-1]
        else:
            net_demand = 0.0

        return AgentObservation(
            step=state.step,
            price=state.price,
            fundamental_value=state.fundamental_value,
            mispricing=mispricing,
            momentum_1=momentum_1,
            momentum_3=momentum_3,
            volatility=volatility,
            net_demand=net_demand,
            shock=state.shock,
        )

    @staticmethod
    def _execute_decision(
        agent: BaseAgent,
        decision,
        execution_price: float,
    ) -> tuple[int, int, bool]:
        if decision.action == TradeAction.HOLD:
            return 0, 0, False

        try:
            agent.apply_trade(decision, execution_price)
        except ValueError as error:
            print(f"[WARN] Failed to execute decision for {agent.agent_id}: {error}")
            return 0, 0, False

        signed_quantity = decision.quantity
        if decision.action == TradeAction.SELL:
            signed_quantity = -decision.quantity

        traded_volume = abs(decision.quantity)
        return signed_quantity, traded_volume, True


    def run_step(self) -> None:
        observation = self._build_observation()

        total_net_demand = 0
        total_volume = 0
        buy_count = 0
        sell_count = 0
        hold_count = 0
        rejected_count = 0

        for agent in self.agents:
            decision = agent.decide(observation)

            if decision.action == TradeAction.BUY:
                buy_count += 1
            elif decision.action == TradeAction.SELL:
                sell_count += 1
            else:
                hold_count += 1

            signed_quantity, traded_volume, executed = self._execute_decision(
                agent,
                decision,
                observation.price,
            )

            if decision.action != TradeAction.HOLD and not executed:
                rejected_count += 1

            total_net_demand += signed_quantity
            total_volume += traded_volume

        old_price = self.market_state.price
        price_change_ratio = self.config.price_impact * total_net_demand
        new_price = max(old_price * (1 + price_change_ratio), 0.01)

        self.market_state.step += 1
        self.market_state.price = new_price
        self.market_state.price_history.append(new_price)
        self.market_state.volume_history.append(total_volume)
        self.market_state.net_demand_history.append(total_net_demand)
        self.market_state.return_history.append((new_price - old_price) / old_price)

        self.buy_count_history.append(buy_count)
        self.sell_count_history.append(sell_count)
        self.hold_count_history.append(hold_count)
        self.rejected_count_history.append(rejected_count)

        if self.market_state.fundamental_value == 0:
            mispricing = 0.0
        else:
            mispricing = (
                self.market_state.price - self.market_state.fundamental_value
            ) / self.market_state.fundamental_value
        self.mispricing_history.append(mispricing)
        self.fundamental_value_history.append(self.market_state.fundamental_value)

        avg_wealth = sum(
            agent.state.cash + agent.state.shares * self.market_state.price
            for agent in self.agents
        ) / len(self.agents)
        self.avg_wealth_history.append(avg_wealth)


    def run_simulation(self) -> None:
        for _ in range(self.config.num_steps):
            self.run_step()

    def get_summary(self) -> dict:
        return {
            "experiment_label": f"{self.config.strategy_type.value}__{self.config.decision_mode.value}",
            "strategy_type": self.config.strategy_type.value,
            "decision_mode": self.config.decision_mode.value,
            "num_agents": len(self.agents),
            "num_steps": self.market_state.step,
            "final_price": self.market_state.price,
            "price_history": self.market_state.price_history,
            "volume_history": self.market_state.volume_history,
            "net_demand_history": self.market_state.net_demand_history,
            "buy_count_history": self.buy_count_history,
            "sell_count_history": self.sell_count_history,
            "hold_count_history": self.hold_count_history,
            "rejected_count_history": self.rejected_count_history,
            "avg_wealth_history": self.avg_wealth_history,
            "mispricing_history": self.mispricing_history,
            "fundamental_value_history": self.fundamental_value_history,
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "cash": agent.state.cash,
                    "shares": agent.state.shares,
                    "avg_cost": agent.state.avg_cost,
                    "wealth": agent.state.cash + agent.state.shares * self.market_state.price,
                }
                for agent in self.agents
            ],
        }

