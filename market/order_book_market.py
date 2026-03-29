"""订单簿市场仿真器。

这个文件负责把 agent、订单簿和实验轮次组织起来。
如果说 `OrderBook` 只回答“订单怎样成交”，
那么 `OrderBookMarket` 回答的是“市场每一轮怎样运转”。
"""

import math
import random

from agents import BaseAgent, FundamentalAgent, MomentumAgent, SpeculatorAgent
from market.order_book import OrderBook
from models import (
    AgentObservation,
    ExperimentConfig,
    LimitOrder,
    MarketState,
    OrderDecision,
    StrategyType,
    TradeAction,
    TradeExecution,
)


class OrderBookMarket:
    """基于限价订单簿的单资产市场。

    这个类承担四类职责：

    - 初始化市场状态、warm-up 历史和 agent
    - 在每一轮内把 agent 随机切成多个 batch
    - 把 agent 决策转成限价单，送入订单簿撮合
    - 结算成交、更新统计量、导出实验摘要
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """初始化市场、订单簿、agent 和统计容器。"""

        self.config = config
        self.rng = random.Random(config.seed)
        self.bootstrap_steps = config.bootstrap_steps
        self.order_sequence = 0
        self.order_book = OrderBook(
            initial_price=config.initial_price,
            price_tick=config.price_tick,
        )

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
        self.agent_lookup = {agent.agent_id: agent for agent in self.agents}
        self.buy_count_history: list[int] = []
        self.sell_count_history: list[int] = []
        self.hold_count_history: list[int] = []
        self.rejected_count_history: list[int] = []
        self.trade_count_history: list[int] = []
        self.batch_count_history: list[int] = []
        self.expired_order_count_history: list[int] = []
        self.open_order_count_history: list[int] = []
        self.best_bid_history: list[float | None] = []
        self.best_ask_history: list[float | None] = []
        self.avg_wealth_history: list[float] = []
        self.sue_history: list[float] = []
        self.reversal_history: list[float] = []
        self.last_actions_by_agent: dict[str, TradeAction] = {
            agent.agent_id: TradeAction.HOLD for agent in self.agents
        }
        self.last_signed_quantity_by_agent = {
            agent.agent_id: 0 for agent in self.agents
        }
        self._bootstrap_market_context()
        self._seed_initial_inventory()
        self.order_book.last_trade_price = self.market_state.price

    def _create_agents(self) -> list[BaseAgent]:
        """根据实验配置实例化一组同策略 agent。"""

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
        """构造正式交易前的 warm-up 历史。"""

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
        """生成 warm-up 收益率序列。"""

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
        """把 warm-up 收益率转换成价格路径。"""

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
        """根据 warm-up 收益率构造一条配套的净需求历史。"""

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
        """为 warm-up 历史生成一个量级合理的成交量序列。"""

        bootstrap_volume: list[int] = []
        base_volume = max(3, self.config.num_agents // 2)

        for signed_demand in bootstrap_net_demand:
            extra_turnover = self.rng.randint(0, base_volume + 2)
            bootstrap_volume.append(
                int(abs(signed_demand) + base_volume + extra_turnover)
            )

        return bootstrap_volume

    def _seed_peer_state_from_bootstrap(self, bootstrap_returns: list[float]) -> None:
        """根据 warm-up 尾部状态，初始化上一轮同伴行为特征。"""

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

    def _seed_initial_inventory(self) -> None:
        """给每个 agent 分配一小笔等值初始持仓。"""

        initial_shares = max(self.config.initial_shares_per_agent, 0)
        if initial_shares == 0:
            return

        reference_price = max(self.market_state.price, self.config.price_tick)
        for agent in self.agents:
            affordable_shares = min(
                initial_shares,
                int(agent.state.cash // reference_price),
            )
            if affordable_shares <= 0:
                continue

            agent.state.cash -= affordable_shares * reference_price
            agent.state.shares = affordable_shares
            agent.state.avg_cost = reference_price

    def _shape_bootstrap_prices(self, bootstrap_prices: list[float]) -> list[float]:
        """对 warm-up 尾部做轻微塑形，让历史更有结构感。"""

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
        """把价格序列转换成收益率序列。"""

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
        """生成供基本面 agent 使用的模拟 SUE 风格信号。"""

        return (
            0.80 * math.sin(2 * math.pi * step / 12)
            + 0.45 * math.sin(2 * math.pi * step / 5 + math.pi / 6)
            + 0.25 * math.cos(2 * math.pi * step / 21 + math.pi / 4)
        )

    def _compute_reversal_score(self) -> float:
        """根据最近价格窗口计算一个简单反转分数。"""

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
        """提取某个 agent 视角下的同伴行为特征。"""

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
        """为单个 agent 构造本轮观测。"""

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

    def _build_agent_batches(self) -> list[list[BaseAgent]]:
        """随机打散 agent，并按固定 batch 大小切分。"""

        shuffled_agents = list(self.agents)
        self.rng.shuffle(shuffled_agents)

        batch_size = max(1, min(self.config.batch_size, len(shuffled_agents)))
        return [
            shuffled_agents[index : index + batch_size]
            for index in range(0, len(shuffled_agents), batch_size)
        ]

    def _align_limit_price(self, raw_price: float, action: TradeAction) -> float:
        """把 agent 生成的原始报价吸附到 tick 网格上。"""

        tick = max(self.config.price_tick, 0.01)
        if action == TradeAction.BUY:
            tick_count = math.ceil(raw_price / tick - 1e-9)
        else:
            tick_count = math.floor(raw_price / tick + 1e-9)
        return max(tick_count * tick, tick)

    def _next_sequence(self) -> int:
        """生成全局递增的订单顺序号。"""

        self.order_sequence += 1
        return self.order_sequence

    def _derive_limit_price(
        self,
        agent: BaseAgent,
        observation: AgentObservation,
        decision: OrderDecision,
    ) -> float:
        """把 agent 决策映射成最终挂单价格。"""

        reference_price = (
            decision.limit_price
            if decision.limit_price is not None and decision.limit_price > 0
            else observation.price
        )
        signal_strength = max(0.0, min(decision.signal_strength, 1.0))
        best_bid = self.order_book.best_bid()
        best_ask = self.order_book.best_ask()
        strategy_cross_bonus = {
            StrategyType.FUNDAMENTAL: -0.05,
            StrategyType.MOMENTUM: 0.00,
            StrategyType.SPECULATOR: 0.12,
        }[agent.strategy_type]
        cross_probability = min(
            max(0.32 + 0.48 * signal_strength + strategy_cross_bonus, 0.18),
            0.90,
        )
        base_offset = max(self.config.price_impact * 0.50, 0.002)
        aggressive_offset = min(
            base_offset * (0.8 + 1.6 * signal_strength) + self.rng.uniform(0.0, 0.002),
            self.config.max_price_offset_ratio,
        )
        passive_offset = min(
            base_offset * (0.35 + 0.85 * (1.0 - 0.5 * signal_strength))
            + self.rng.uniform(0.0, 0.0015),
            self.config.max_price_offset_ratio,
        )

        if decision.action == TradeAction.BUY:
            if self.rng.random() < cross_probability:
                anchor_price = max(reference_price, best_ask or reference_price)
                raw_price = anchor_price * (1 + aggressive_offset)
            else:
                anchor_price = min(reference_price, best_bid or reference_price)
                raw_price = anchor_price * (1 - passive_offset)

            if decision.quantity > 0:
                max_affordable_price = agent.state.cash / decision.quantity
                raw_price = min(raw_price, max_affordable_price)

            return self._align_limit_price(raw_price, TradeAction.BUY)

        anchor_price = min(reference_price, best_bid or reference_price)
        if self.rng.random() < cross_probability:
            raw_price = anchor_price * (1 - aggressive_offset)
        else:
            anchor_price = max(reference_price, best_ask or reference_price)
            raw_price = anchor_price * (1 + passive_offset)

        return self._align_limit_price(raw_price, TradeAction.SELL)

    def _create_limit_order(
        self,
        agent: BaseAgent,
        observation: AgentObservation,
        decision: OrderDecision,
        step: int,
        batch: int,
    ) -> LimitOrder | None:
        """把决策对象转换成真正可入簿的限价单。"""

        if decision.action == TradeAction.HOLD or decision.quantity <= 0:
            return None

        limit_price = self._derive_limit_price(agent, observation, decision)
        if limit_price <= 0:
            return None

        if decision.action == TradeAction.BUY:
            max_affordable_quantity = int(agent.state.cash // limit_price)
            quantity = min(decision.quantity, max_affordable_quantity)
        else:
            quantity = min(decision.quantity, agent.state.shares)

        if quantity <= 0:
            return None

        sequence = self._next_sequence()
        return LimitOrder(
            order_id=f"{agent.agent_id}_order_{sequence}",
            agent_id=agent.agent_id,
            action=decision.action,
            quantity=quantity,
            limit_price=limit_price,
            submitted_step=step,
            submitted_batch=batch,
            sequence=sequence,
            reason=decision.reason,
            signal_strength=decision.signal_strength,
        )

    @staticmethod
    def _signed_quantity(action: TradeAction, quantity: int) -> int:
        """把买卖方向转换成带符号的数量。"""

        if action == TradeAction.BUY:
            return quantity
        if action == TradeAction.SELL:
            return -quantity
        return 0

    @staticmethod
    def _apply_execution(
        agent: BaseAgent,
        action: TradeAction,
        quantity: int,
        execution_price: float,
    ) -> None:
        """把一笔成交真正记入 agent 账户。"""

        execution_decision = OrderDecision(
            agent_id=agent.agent_id,
            action=action,
            quantity=quantity,
            limit_price=execution_price,
            reason="matched in the order book",
            signal_strength=0.0,
        )
        agent.apply_trade(execution_decision, execution_price)

    def _inject_batch_liquidity(self, step: int, batch: int) -> list[str]:
        """向当前 batch 注入一层很薄的背景流动性。"""

        reference_price = max(self.order_book.reference_price(), self.config.price_tick)
        liquidity_depth = max(2, self.config.num_agents // 3)
        spread_ratio = 0.003 + self.rng.uniform(0.0, 0.002)

        bid_sequence = self._next_sequence()
        ask_sequence = self._next_sequence()
        bid_agent_id = f"external_bid_{step}_{batch}"
        ask_agent_id = f"external_ask_{step}_{batch}"

        bid_order = LimitOrder(
            order_id=f"{bid_agent_id}_order",
            agent_id=bid_agent_id,
            action=TradeAction.BUY,
            quantity=liquidity_depth,
            limit_price=self._align_limit_price(
                reference_price * (1 - spread_ratio),
                TradeAction.BUY,
            ),
            submitted_step=step,
            submitted_batch=batch,
            sequence=bid_sequence,
            reason="background liquidity",
            signal_strength=0.0,
        )
        ask_order = LimitOrder(
            order_id=f"{ask_agent_id}_order",
            agent_id=ask_agent_id,
            action=TradeAction.SELL,
            quantity=liquidity_depth,
            limit_price=self._align_limit_price(
                reference_price * (1 + spread_ratio),
                TradeAction.SELL,
            ),
            submitted_step=step,
            submitted_batch=batch,
            sequence=ask_sequence,
            reason="background liquidity",
            signal_strength=0.0,
        )

        self.order_book.add_order(bid_order)
        self.order_book.add_order(ask_order)
        return [bid_agent_id, ask_agent_id]

    def _settle_trade(self, trade: TradeExecution) -> None:
        """对一笔成交执行资金和持仓结算。"""

        buyer = self.agent_lookup.get(trade.buyer_id)
        seller = self.agent_lookup.get(trade.seller_id)

        if buyer is not None:
            self._apply_execution(
                buyer,
                TradeAction.BUY,
                trade.quantity,
                trade.price,
            )
        if seller is not None:
            self._apply_execution(
                seller,
                TradeAction.SELL,
                trade.quantity,
                trade.price,
            )

    @staticmethod
    def _serialize_order(order: LimitOrder) -> dict:
        """把订单对象转成可写入摘要文件的字典。"""

        return {
            "order_id": order.order_id,
            "agent_id": order.agent_id,
            "action": order.action.value,
            "quantity": order.quantity,
            "limit_price": order.limit_price,
            "submitted_step": order.submitted_step,
            "submitted_batch": order.submitted_batch,
            "sequence": order.sequence,
            "reason": order.reason,
            "signal_strength": order.signal_strength,
        }

    @staticmethod
    def _serialize_trade(trade: TradeExecution) -> dict:
        """把成交对象转成可写入摘要文件的字典。"""

        return {
            "trade_id": trade.trade_id,
            "buy_order_id": trade.buy_order_id,
            "sell_order_id": trade.sell_order_id,
            "price": trade.price,
            "quantity": trade.quantity,
            "step": trade.step,
            "batch": trade.batch,
            "sequence": trade.sequence,
            "buyer_id": trade.buyer_id,
            "seller_id": trade.seller_id,
        }

    def run_step(self) -> None:
        """推进市场一轮。"""

        opening_price = self.market_state.price
        step_sue_signal = self._generate_sue_signal(self.market_state.step)
        step_reversal_score = self._compute_reversal_score()
        current_step = self.market_state.step + 1

        total_net_demand = 0
        total_volume = 0
        trade_count = 0
        buy_count = 0
        sell_count = 0
        hold_count = 0
        rejected_count = 0
        current_actions_by_agent: dict[str, TradeAction] = {
            agent.agent_id: TradeAction.HOLD for agent in self.agents
        }
        current_signed_quantity_by_agent = {
            agent.agent_id: 0 for agent in self.agents
        }

        expired_orders = self.order_book.clear_expired_orders(
            current_step=current_step,
            max_age_steps=self.config.max_order_age_steps,
        )
        agent_batches = self._build_agent_batches()

        for batch_index, batch_agents in enumerate(agent_batches, start=1):
            for agent in batch_agents:
                self.order_book.cancel_agent_order(agent.agent_id)

            liquidity_agent_ids = self._inject_batch_liquidity(
                step=current_step,
                batch=batch_index,
            )
            self.market_state.price = self.order_book.reference_price()
            batch_price = self.market_state.price
            batch_orders: list[LimitOrder] = []

            for agent in batch_agents:
                self.market_state.price = batch_price
                observation = self._build_observation(agent)
                decision = agent.decide(observation)

                if decision.action == TradeAction.HOLD:
                    hold_count += 1
                    continue

                order = self._create_limit_order(
                    agent=agent,
                    observation=observation,
                    decision=decision,
                    step=current_step,
                    batch=batch_index,
                )
                if order is None:
                    rejected_count += 1
                    hold_count += 1
                    continue

                batch_orders.append(order)
                signed_quantity = self._signed_quantity(order.action, order.quantity)
                total_net_demand += signed_quantity
                current_actions_by_agent[agent.agent_id] = order.action
                current_signed_quantity_by_agent[agent.agent_id] = signed_quantity

                if order.action == TradeAction.BUY:
                    buy_count += 1
                else:
                    sell_count += 1

            for order in batch_orders:
                self.order_book.add_order(order)

            batch_trades = self.order_book.match_orders(
                step=current_step,
                batch=batch_index,
            )
            for trade in batch_trades:
                self._settle_trade(trade)
                total_volume += trade.quantity
            trade_count += len(batch_trades)

            for liquidity_agent_id in liquidity_agent_ids:
                self.order_book.cancel_agent_order(liquidity_agent_id)
            self.market_state.price = self.order_book.reference_price()

        closing_price = self.order_book.reference_price()

        self.market_state.step += 1
        self.market_state.price = closing_price
        self.market_state.price_history.append(closing_price)
        self.market_state.volume_history.append(total_volume)
        self.market_state.net_demand_history.append(total_net_demand)
        self.market_state.return_history.append(
            (closing_price - opening_price) / opening_price if opening_price else 0.0
        )

        self.buy_count_history.append(buy_count)
        self.sell_count_history.append(sell_count)
        self.hold_count_history.append(hold_count)
        self.rejected_count_history.append(rejected_count)
        self.trade_count_history.append(trade_count)
        self.batch_count_history.append(len(agent_batches))
        self.expired_order_count_history.append(expired_orders)
        self.open_order_count_history.append(self.order_book.order_count())
        self.best_bid_history.append(self.order_book.best_bid())
        self.best_ask_history.append(self.order_book.best_ask())
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
        """连续运行多轮市场。"""

        for _ in range(self.config.num_steps):
            self.run_step()

    def get_summary(self) -> dict:
        """导出实验摘要。"""

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
            "initial_shares_per_agent": self.config.initial_shares_per_agent,
            "batch_size": self.config.batch_size,
            "max_order_age_steps": self.config.max_order_age_steps,
            "price_tick": self.config.price_tick,
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
            "trade_count_history": self.trade_count_history,
            "batch_count_history": self.batch_count_history,
            "expired_order_count_history": self.expired_order_count_history,
            "open_order_count_history": self.open_order_count_history,
            "best_bid_history": self.best_bid_history,
            "best_ask_history": self.best_ask_history,
            "avg_wealth_history": self.avg_wealth_history,
            "sue_history": self.sue_history,
            "reversal_history": self.reversal_history,
            "trade_history": [
                self._serialize_trade(trade) for trade in self.order_book.trades
            ],
            "open_buy_orders": [
                self._serialize_order(order) for order in self.order_book.buy_orders
            ],
            "open_sell_orders": [
                self._serialize_order(order) for order in self.order_book.sell_orders
            ],
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
