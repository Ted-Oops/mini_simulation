import math
import random

from agents import BaseAgent, FundamentalAgent, MomentumAgent, SpeculatorAgent
from models import (
    ExperimentConfig,
    MarketState,
    StrategyType,
    AgentObservation,
    OrderDecision,
    TradeAction,
)


class SimpleMarket:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.bootstrap_steps = config.bootstrap_steps

        self.market_state = MarketState(
            step=0,
            price=config.initial_price,
            price_history=[config.initial_price],
            return_history=[],
            volume_history=[],
            net_demand_history=[],
            shock=0.0,
        )

        self.agents = self._create_agents()
        self.buy_count_history = []
        self.sell_count_history = []
        self.hold_count_history = []
        self.rejected_count_history = []
        self.avg_wealth_history = []
        self.sue_history = []
        self.reversal_history = []
        self.last_actions_by_agent = {
            agent.agent_id: TradeAction.HOLD for agent in self.agents
        }
        self.last_signed_quantity_by_agent = {
            agent.agent_id: 0 for agent in self.agents
        }
        self._bootstrap_market_context()

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
                    initial_cash=self.config.initial_cash,
                )
            )
        return agents

    def _bootstrap_market_context(self) -> None:
        if self.bootstrap_steps <= 0:
            return

        bootstrap_returns = self._generate_bootstrap_returns(self.bootstrap_steps)
        bootstrap_prices = self._build_bootstrap_prices(bootstrap_returns)
        bootstrap_prices = self._shape_bootstrap_prices(bootstrap_prices)
        bootstrap_returns = self._returns_from_prices(bootstrap_prices)
        bootstrap_net_demand = self._generate_bootstrap_net_demand(bootstrap_returns)
        bootstrap_volume = self._generate_bootstrap_volume(bootstrap_net_demand)

        self.market_state.price_history = bootstrap_prices
        self.market_state.return_history = bootstrap_returns
        self.market_state.net_demand_history = bootstrap_net_demand
        self.market_state.volume_history = bootstrap_volume
        self.market_state.price = bootstrap_prices[-1]

        self._seed_peer_state_from_bootstrap(bootstrap_returns)

    def _generate_bootstrap_returns(self, num_steps: int) -> list[float]:
        bootstrap_returns: list[float] = []
        previous_return = 0.0

        for step in range(num_steps):
            cyclical_component = 0.009 * math.sin(
                2 * math.pi * (step + 1) / 6.5 + 0.55
            ) + 0.006 * math.cos(2 * math.pi * (step + 1) / 11.0 + 1.10)
            persistence_component = 0.28 * previous_return
            noise_component = self.rng.uniform(-0.012, 0.012)
            burst_component = 0.0

            if step % 7 in (2, 5):
                burst_component = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(
                    0.004, 0.014
                )

            period_return = (
                cyclical_component
                + persistence_component
                + noise_component
                + burst_component
            )
            period_return = max(min(period_return, 0.032), -0.032)

            bootstrap_returns.append(period_return)
            previous_return = period_return

        return bootstrap_returns

    def _build_bootstrap_prices(self, bootstrap_returns: list[float]) -> list[float]:
        start_price = self.config.initial_price * (0.94 + 0.05 * self.rng.random())
        bootstrap_prices = [start_price]

        for period_return in bootstrap_returns:
            next_price = max(bootstrap_prices[-1] * (1 + period_return), 0.01)
            bootstrap_prices.append(next_price)

        scaling_factor = self.config.initial_price / bootstrap_prices[-1]
        return [max(price * scaling_factor, 0.01) for price in bootstrap_prices]

    def _generate_bootstrap_net_demand(
        self, bootstrap_returns: list[float]
    ) -> list[float]:
        bootstrap_net_demand: list[float] = []
        max_abs_demand = max(2, self.config.num_agents - 1)

        for period_return in bootstrap_returns:
            noisy_signal = (period_return / 0.012) * (self.config.num_agents / 3)
            noisy_signal += self.rng.uniform(-1.5, 1.5)
            signed_demand = int(round(noisy_signal))
            signed_demand = max(min(signed_demand, max_abs_demand), -max_abs_demand)

            if signed_demand == 0 and abs(period_return) > 0.008:
                signed_demand = 1 if period_return > 0 else -1

            bootstrap_net_demand.append(signed_demand)

        return bootstrap_net_demand

    def _generate_bootstrap_volume(
        self, bootstrap_net_demand: list[float]
    ) -> list[int]:
        bootstrap_volume: list[int] = []
        base_volume = max(3, self.config.num_agents // 2)

        for signed_demand in bootstrap_net_demand:
            extra_turnover = self.rng.randint(0, base_volume + 2)
            bootstrap_volume.append(
                int(abs(signed_demand) + base_volume + extra_turnover)
            )

        return bootstrap_volume

    def _seed_peer_state_from_bootstrap(self, bootstrap_returns: list[float]) -> None:
        if not bootstrap_returns:
            return

        recent_mean_return = sum(bootstrap_returns[-3:]) / min(
            3, len(bootstrap_returns)
        )
        current_sentiment = 0.6 * bootstrap_returns[-1] + 0.4 * recent_mean_return

        if current_sentiment > 0.008:
            buy_probability, sell_probability = 0.45, 0.20
        elif current_sentiment < -0.008:
            buy_probability, sell_probability = 0.20, 0.45
        else:
            buy_probability, sell_probability = 0.34, 0.30

        seeded_actions: dict[str, TradeAction] = {}
        seeded_quantities: dict[str, int] = {}

        for agent in self.agents:
            draw = self.rng.random()
            if draw < buy_probability:
                action = TradeAction.BUY
                signed_quantity = 1
            elif draw < buy_probability + sell_probability:
                action = TradeAction.SELL
                signed_quantity = -1
            else:
                action = TradeAction.HOLD
                signed_quantity = 0

            seeded_actions[agent.agent_id] = action
            seeded_quantities[agent.agent_id] = signed_quantity

        unique_actions = set(seeded_actions.values())
        if len(unique_actions) == 1 and len(self.agents) >= 2:
            first_agent_id = self.agents[0].agent_id
            second_agent_id = self.agents[1].agent_id
            seeded_actions[first_agent_id] = TradeAction.BUY
            seeded_quantities[first_agent_id] = 1
            seeded_actions[second_agent_id] = TradeAction.SELL
            seeded_quantities[second_agent_id] = -1

        self.last_actions_by_agent = seeded_actions
        self.last_signed_quantity_by_agent = seeded_quantities

    def _shape_bootstrap_prices(self, bootstrap_prices: list[float]) -> list[float]:
        if len(bootstrap_prices) < 6:
            return bootstrap_prices

        final_price = self.config.initial_price
        tail_template = [
            1.145 + self.rng.uniform(-0.010, 0.010),
            1.085 + self.rng.uniform(-0.008, 0.008),
            0.945 + self.rng.uniform(-0.008, 0.008),
            0.972 + self.rng.uniform(-0.006, 0.006),
            1.000,
        ]
        shaped_tail = [
            max(final_price * multiplier, 0.01) for multiplier in tail_template
        ]
        return bootstrap_prices[:-5] + shaped_tail

    @staticmethod
    def _returns_from_prices(bootstrap_prices: list[float]) -> list[float]:
        bootstrap_returns: list[float] = []
        for index in range(len(bootstrap_prices) - 1):
            current_price = bootstrap_prices[index]
            next_price = bootstrap_prices[index + 1]
            if current_price <= 0:
                bootstrap_returns.append(0.0)
            else:
                bootstrap_returns.append((next_price - current_price) / current_price)
        return bootstrap_returns

    @staticmethod
    def _generate_sue_signal(step: int) -> float:
        return (
            0.80 * math.sin(2 * math.pi * step / 12)
            + 0.45 * math.sin(2 * math.pi * step / 5 + math.pi / 6)
            + 0.25 * math.cos(2 * math.pi * step / 21 + math.pi / 4)
        )

    def _compute_reversal_score(self) -> float:
        prices = self.market_state.price_history
        if len(prices) < 3:
            return 0.0

        window = prices[-5:] if len(prices) >= 5 else prices
        mean_price = sum(window) / len(window)
        if mean_price == 0:
            return 0.0
        return (mean_price - self.market_state.price) / mean_price

    def _get_peer_action_features(
        self, agent_id: str
    ) -> tuple[float, float, float, float]:
        peer_items = [
            (peer_id, action)
            for peer_id, action in self.last_actions_by_agent.items()
            if peer_id != agent_id
        ]
        if not peer_items:
            return 0.0, 0.0, 1.0, 0.0

        peer_count = len(peer_items)
        buy_ratio = (
            sum(action == TradeAction.BUY for _, action in peer_items) / peer_count
        )
        sell_ratio = (
            sum(action == TradeAction.SELL for _, action in peer_items) / peer_count
        )
        hold_ratio = (
            sum(action == TradeAction.HOLD for _, action in peer_items) / peer_count
        )
        peer_net_demand = sum(
            self.last_signed_quantity_by_agent.get(peer_id, 0)
            for peer_id, _ in peer_items
        )
        return buy_ratio, sell_ratio, hold_ratio, peer_net_demand

    def _build_observation(self, agent: BaseAgent) -> AgentObservation:
        state = self.market_state
        sue_signal = self._generate_sue_signal(state.step)
        reversal_score = self._compute_reversal_score()
        if agent.strategy_type == StrategyType.SPECULATOR:
            (
                peer_buy_ratio,
                peer_sell_ratio,
                peer_hold_ratio,
                peer_net_demand,
            ) = self._get_peer_action_features(agent.agent_id)
        else:
            peer_buy_ratio = 0.0
            peer_sell_ratio = 0.0
            peer_hold_ratio = 0.0
            peer_net_demand = 0.0

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
            squared_diffs = [(ret - mean_return) ** 2 for ret in state.return_history]
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
            sue_signal=sue_signal,
            momentum_1=momentum_1,
            momentum_3=momentum_3,
            reversal_score=reversal_score,
            volatility=volatility,
            net_demand=net_demand,
            peer_buy_ratio=peer_buy_ratio,
            peer_sell_ratio=peer_sell_ratio,
            peer_hold_ratio=peer_hold_ratio,
            peer_net_demand=peer_net_demand,
            shock=state.shock,
        )

    @staticmethod
    def _execute_decision(
        agent: BaseAgent,
        decision: OrderDecision,
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
        total_net_demand = 0
        total_volume = 0
        buy_count = 0
        sell_count = 0
        hold_count = 0
        rejected_count = 0
        current_actions_by_agent = {}
        current_signed_quantity_by_agent = {}
        step_sue_signal = self._generate_sue_signal(self.market_state.step)
        step_reversal_score = self._compute_reversal_score()

        for agent in self.agents:
            observation = self._build_observation(agent)
            decision = agent.decide(observation)

            signed_quantity, traded_volume, executed = self._execute_decision(
                agent,
                decision,
                observation.price,
            )

            if decision.action != TradeAction.HOLD and not executed:
                rejected_count += 1

            realized_action = decision.action if executed else TradeAction.HOLD
            current_actions_by_agent[agent.agent_id] = realized_action
            current_signed_quantity_by_agent[agent.agent_id] = signed_quantity

            if realized_action == TradeAction.BUY:
                buy_count += 1
            elif realized_action == TradeAction.SELL:
                sell_count += 1
            else:
                hold_count += 1

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
        self.sue_history.append(step_sue_signal)
        self.reversal_history.append(step_reversal_score)

        avg_wealth = sum(
            agent.state.cash + agent.state.shares * self.market_state.price
            for agent in self.agents
        ) / len(self.agents)
        self.avg_wealth_history.append(avg_wealth)
        self.last_actions_by_agent = current_actions_by_agent
        self.last_signed_quantity_by_agent = current_signed_quantity_by_agent

    def run_simulation(self) -> None:
        for _ in range(self.config.num_steps):
            self.run_step()

    def get_summary(self) -> dict:
        simulation_price_history = self.market_state.price_history[
            self.bootstrap_steps :
        ]
        simulation_volume_history = self.market_state.volume_history[
            self.bootstrap_steps :
        ]
        simulation_net_demand_history = self.market_state.net_demand_history[
            self.bootstrap_steps :
        ]

        return {
            "experiment_label": f"{self.config.strategy_type.value}__{self.config.decision_mode.value}",
            "strategy_type": self.config.strategy_type.value,
            "decision_mode": self.config.decision_mode.value,
            "num_agents": len(self.agents),
            "num_steps": self.market_state.step,
            "bootstrap_steps": self.bootstrap_steps,
            "final_price": self.market_state.price,
            "price_history": simulation_price_history,
            "volume_history": simulation_volume_history,
            "net_demand_history": simulation_net_demand_history,
            "bootstrap_price_history": self.market_state.price_history[
                : self.bootstrap_steps + 1
            ],
            "bootstrap_return_history": self.market_state.return_history[
                : self.bootstrap_steps
            ],
            "bootstrap_volume_history": self.market_state.volume_history[
                : self.bootstrap_steps
            ],
            "bootstrap_net_demand_history": self.market_state.net_demand_history[
                : self.bootstrap_steps
            ],
            "buy_count_history": self.buy_count_history,
            "sell_count_history": self.sell_count_history,
            "hold_count_history": self.hold_count_history,
            "rejected_count_history": self.rejected_count_history,
            "avg_wealth_history": self.avg_wealth_history,
            "sue_history": self.sue_history,
            "reversal_history": self.reversal_history,
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "cash": agent.state.cash,
                    "shares": agent.state.shares,
                    "avg_cost": agent.state.avg_cost,
                    "wealth": agent.state.cash
                    + agent.state.shares * self.market_state.price,
                }
                for agent in self.agents
            ],
        }
