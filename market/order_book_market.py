"""订单簿市场仿真器。

这个文件保留市场主循环。Warm-up、观测构造、订单生成、
背景流动性、batch 决策和 summary 导出分散在同包的辅助模块中。
"""

import random

from agents import BaseAgent
from market.agent_factory import create_agents
from market.batching import build_agent_batches, decide_batch
from market.bootstrap import bootstrap_market_context, seed_initial_inventory
from market.liquidity import BackgroundLiquidityManager
from market.observation import (
    build_observation,
    compute_reversal_score,
    generate_sue_signal,
)
from market.order_book import OrderBook
from market.orders import create_limit_order, settle_trade, signed_quantity
from market.summary import MarketHistories, build_summary
from models import (
    ExperimentConfig,
    LimitOrder,
    MarketState,
    TradeAction,
)


class OrderBookMarket:
    """基于限价订单簿的单资产市场。"""

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
            mark_price=config.initial_price,
            price_history=[config.initial_price],
            mark_price_history=[config.initial_price],
            return_history=[],
            volume_history=[],
            net_demand_history=[],
            shock=0.0,
        )

        self.agents = create_agents(
            strategy_type=config.strategy_type,
            decision_mode=config.decision_mode,
            num_agents=config.num_agents,
            initial_cash=config.initial_cash,
        )
        self.agent_lookup: dict[str, BaseAgent] = {
            agent.agent_id: agent for agent in self.agents
        }
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

        seeded_peer_state = bootstrap_market_context(
            config=self.config,
            rng=self.rng,
            market_state=self.market_state,
            agents=self.agents,
        )
        if seeded_peer_state is not None:
            (
                self.last_actions_by_agent,
                self.last_signed_quantity_by_agent,
            ) = seeded_peer_state

        seed_initial_inventory(
            config=self.config,
            market_state=self.market_state,
            agents=self.agents,
        )
        self.order_book.last_trade_price = self.market_state.price
        self.background_liquidity = BackgroundLiquidityManager(
            config=self.config,
            rng=self.rng,
            order_book=self.order_book,
            next_sequence=self._next_sequence,
            anchor_price=self.market_state.price,
        )

    def _next_sequence(self) -> int:
        """生成全局递增的订单顺序号。"""

        self.order_sequence += 1
        return self.order_sequence

    def _refresh_mark_price(self) -> float:
        """根据当前盘口刷新 quote-based mark price。"""

        self.market_state.mark_price = self.order_book.mark_price()
        return self.market_state.mark_price

    def run_step(self) -> None:
        """推进市场一轮。"""

        opening_price = self.market_state.price
        step_sue_signal = generate_sue_signal(self.market_state.step)
        step_reversal_score = compute_reversal_score(self.market_state)
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
        agent_batches = build_agent_batches(
            agents=self.agents,
            rng=self.rng,
            batch_size=self.config.batch_size,
        )

        for batch_index, batch_agents in enumerate(agent_batches, start=1):
            for agent in batch_agents:
                self.order_book.cancel_agent_order(agent.agent_id)

            self.background_liquidity.maintain(
                step=current_step,
                batch=batch_index,
                reference_price=self.market_state.price,
            )
            self._refresh_mark_price()
            batch_orders: list[LimitOrder] = []
            batch_observations = [
                build_observation(
                    market_state=self.market_state,
                    agent=agent,
                    last_actions_by_agent=self.last_actions_by_agent,
                    last_signed_quantity_by_agent=self.last_signed_quantity_by_agent,
                )
                for agent in batch_agents
            ]
            batch_decisions = decide_batch(
                decision_mode=self.config.decision_mode,
                batch_size=self.config.batch_size,
                batch_agents=batch_agents,
                observations=batch_observations,
            )

            for agent, observation, decision in zip(
                batch_agents,
                batch_observations,
                batch_decisions,
            ):
                if decision.action == TradeAction.HOLD:
                    hold_count += 1
                    continue

                order, self.order_sequence = create_limit_order(
                    config=self.config,
                    rng=self.rng,
                    order_book=self.order_book,
                    order_sequence=self.order_sequence,
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
                order_signed_quantity = signed_quantity(order.action, order.quantity)
                total_net_demand += order_signed_quantity
                current_actions_by_agent[agent.agent_id] = order.action
                current_signed_quantity_by_agent[agent.agent_id] = (
                    order_signed_quantity
                )

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
                settle_trade(self.agent_lookup, trade)
                total_volume += trade.quantity
            trade_count += len(batch_trades)
            if batch_trades:
                self.market_state.price = batch_trades[-1].price
                self.background_liquidity.realign_after_trades(
                    step=current_step,
                    batch=batch_index,
                    reference_price=self.market_state.price,
                )

            self._refresh_mark_price()

        closing_price = self.market_state.price
        closing_mark_price = self._refresh_mark_price()

        self.market_state.step += 1
        self.market_state.price = closing_price
        self.market_state.mark_price = closing_mark_price
        self.market_state.price_history.append(closing_price)
        self.market_state.mark_price_history.append(closing_mark_price)
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

        return build_summary(
            config=self.config,
            bootstrap_steps=self.bootstrap_steps,
            market_state=self.market_state,
            agents=self.agents,
            order_book=self.order_book,
            histories=MarketHistories(
                buy_count_history=self.buy_count_history,
                sell_count_history=self.sell_count_history,
                hold_count_history=self.hold_count_history,
                rejected_count_history=self.rejected_count_history,
                trade_count_history=self.trade_count_history,
                batch_count_history=self.batch_count_history,
                expired_order_count_history=self.expired_order_count_history,
                open_order_count_history=self.open_order_count_history,
                best_bid_history=self.best_bid_history,
                best_ask_history=self.best_ask_history,
                avg_wealth_history=self.avg_wealth_history,
                sue_history=self.sue_history,
                reversal_history=self.reversal_history,
            ),
        )
