import math
import random
from collections import Counter
from dataclasses import asdict

import numpy as np

from market.config import MarketConfig
from market.order_book import OrderBook


def lambert_w_from_log(log_z):
    
    if log_z < -36:
        return math.exp(log_z)
    w = max(0.1, log_z - math.log(log_z)) if log_z > 1 else math.exp(log_z)
    for _ in range(12):
        error = w + math.log(w) - log_z
        delta = error * w / (1 + w)
        candidate = w - delta
        if candidate <= 0:
            candidate = w / 2
        if abs(candidate - w) < 2e-14 * max(1.0, w):
            return candidate
        w = candidate
    return w


def price_bounds(log_forecast, av, shares, cash):
    
    if shares <= 0:
        return math.exp(log_forecast - av * cash), math.exp(log_forecast), math.exp(log_forecast)
    log_c = math.log(av * shares) + log_forecast
    satisfaction = math.exp(log_forecast - lambert_w_from_log(log_c))
    lower_log = log_forecast - av * cash
    minimum = math.exp(lower_log - lambert_w_from_log(log_c - av * cash))
    return minimum, satisfaction, math.exp(log_forecast)


def simulate(config: MarketConfig):
    parameters = random.Random(config.seed + 200003)
    activations = random.Random(config.seed + 300007)
    fundamentals = random.Random(config.seed + 400009)
    noises = random.Random(config.seed + 500009)
    quotes = random.Random(config.seed + 600011)
    initialization = random.Random(config.seed + 700001)
    histories = random.Random(config.seed + 800011)
    weights, horizons, risks = [], [], []
    for _ in range(config.agents):
        wf, wc, wn = [parameters.expovariate(1 / scale) if scale else 0.0 for scale in
                      (config.fundamental_weight, config.chartist_weight, config.noise_weight)]
        weights.append((wf, wc, wn))
        
        trait_scale = wf + wc + wn if config.trait_weights == "normalized" else 1.0
        ratio = (1 + wf / trait_scale) / (1 + wc / trait_scale)
        horizons.append(math.ceil(config.reference_horizon * ratio))
        risks.append(config.risk_aversion * ratio)
    cash = [initialization.uniform(0, config.max_initial_cash) for _ in horizons]
    shares = [initialization.uniform(0, config.max_initial_shares) for _ in horizons]
    if config.quantity_mode == "integer":
        shares = [float(int(s)) for s in shares]
    initial_cash, initial_shares = math.fsum(cash), math.fsum(shares)
    book = OrderBook(cash, shares, config.tick)
    warmup_length = max(horizons) + 1
    warmup_sigma = {"noise": config.noise_std, "fundamental": config.fundamental_sigma, "flat": 0.0}[config.warmup]
    past_returns = [histories.gauss(0, warmup_sigma) for _ in range(warmup_length)]
    sums, squares = [0.0], [0.0]
    for value in past_returns:
        sums.append(sums[-1] + value)
        squares.append(squares[-1] + value * value)
    fields = ("price", "fundamental", "return", "executed_volume", "submitted_volume", "side", "market_order",
              "limit_volume", "placement_mid", "placement_best", "bid_gap", "ask_gap", "opposite_gap", "spread",
              "active_orders", "two_sided", "variance", "agent", "log_forecast", "forecast_clipped")
    total_events = config.initialization_steps + config.steps
    data = {name: np.zeros(total_events, dtype=np.float64) for name in fields}
    for name in ("placement_mid", "placement_best", "bid_gap", "ask_gap", "opposite_gap", "spread"):
        data[name].fill(np.nan)
    price = fundamental = config.initial_price
    floor_count = forecast_errors = 0
    profile = {"buy": Counter(), "sell": Counter()}
    profile_samples = 0
    for event in range(total_events):
        step = event + 1
        clock_scale = 1 / config.agents if config.fundamental_clock == "population" else 1.0
        if event >= config.initialization_steps:
            fundamental *= math.exp(-0.5 * config.fundamental_sigma ** 2 * clock_scale
                                    + config.fundamental_sigma * math.sqrt(clock_scale) * fundamentals.gauss(0, 1))
        book.expire(step)
        agent = activations.randrange(config.agents)
        if config.order_policy == "replace":
            book.cancel_owner(agent)
        bid, ask = book.best(1), book.best(-1)
        bid_gap, ask_gap = book.gap(1), book.gap(-1)
        midpoint = (bid + ask) / 2 if bid is not None and ask is not None else price
        horizon = horizons[agent]
        length = len(sums) - 1
        trend = (sums[-1] - sums[length - horizon]) / horizon
        raw_variance = (squares[-1] - squares[length - horizon]) / horizon - trend * trend
        variance = max(raw_variance, config.variance_floor)
        floor_count += raw_variance < config.variance_floor
        wf, wc, wn = weights[agent]
        fundamental_horizon = horizon if config.fundamental_horizon_mode == "individual" else config.fundamental_horizon
        expected = (wf * math.log(fundamental / price) / fundamental_horizon
                    + wc * trend + wn * noises.gauss(0, config.noise_std)) / (wf + wc + wn)
        log_forecast = math.log(price) + expected * horizon
        
        if not -700 < log_forecast < 700:
            raise FloatingPointError(f"预测价格超出可计算范围，event={event}, log_forecast={log_forecast}")
        av = risks[agent] * variance
        minimum, satisfaction, maximum = price_bounds(log_forecast, av, shares[agent], book.available_cash(agent))
        lower_tick = max(1, math.ceil(minimum / config.tick - 1e-9))
        upper_tick = math.floor(maximum / config.tick + 1e-9)
        side = market_order = 0
        submitted = executed = residual = 0.0
        last_trade = None
        placement_best = placement_mid = math.nan
        if upper_tick >= lower_tick or config.grid_rule == "sample_then_round":
            if config.grid_rule == "sample_then_round":
                sampled = quotes.uniform(minimum, maximum)
                ticks = max(1, math.floor(sampled / config.tick + 0.5))
            else:
                ticks = quotes.randrange(lower_tick, upper_tick + 1)
            quote = ticks * config.tick
            side = 1 if quote < satisfaction else -1 if quote > satisfaction else 0
            market_order = bool((side == 1 and ask is not None and quote >= ask)
                                or (side == -1 and bid is not None and quote <= bid))
            demand_price = ask if side == 1 and market_order else bid if side == -1 and market_order else quote
            desired = max(0.0, (log_forecast - math.log(demand_price)) / (av * demand_price))
            submitted = max(0.0, desired - shares[agent] if side == 1 else shares[agent] - desired if side == -1 else 0.0)
            if config.quantity_mode == "integer":
                submitted = float(int(submitted))
            if side:
                last_trade, executed, residual = book.submit(agent, side, ticks, submitted, step, horizon,
                                                              integer=config.quantity_mode == "integer")
            if residual > 1e-10:
                placement_mid = abs(quote - midpoint)
                best_same = bid if side == 1 else ask
                placement_best = abs(quote - best_same) if best_same is not None else math.nan
        previous = price
        if last_trade is not None:
            price = last_trade
        else:
            post_bid, post_ask = book.best(1), book.best(-1)
            if post_bid is not None and post_ask is not None:
                price = (post_bid + post_ask) / 2
        ret = math.log(price / previous)
        if abs(ret) < 1e-14:
            ret = 0.0
        sums.append(sums[-1] + ret)
        squares.append(squares[-1] + ret * ret)
        values = {"price": price, "fundamental": fundamental, "return": ret, "executed_volume": executed,
                  "submitted_volume": submitted, "side": side, "market_order": market_order, "limit_volume": residual,
                  "placement_mid": placement_mid, "placement_best": placement_best, "bid_gap": bid_gap,
                  "ask_gap": ask_gap, "opposite_gap": ask_gap if side == 1 else bid_gap if side == -1 else None,
                  "spread": ask - bid if bid is not None and ask is not None else None,
                  "active_orders": len(book.orders), "two_sided": bid is not None and ask is not None,
                  "variance": variance, "agent": agent, "log_forecast": log_forecast, "forecast_clipped": 0}
        for name, value in values.items():
            data[name][event] = math.nan if value is None else value
        if step > config.initialization_steps + config.burn_in and event % config.profile_stride == 0:
            profile_samples += 1
            current_bid, current_ask = book.best(1), book.best(-1)
            center = (current_bid + current_ask) / 2 if current_bid is not None and current_ask is not None else price
            for order in book.orders.values():
                distance = round(abs(order.ticks * config.tick - center) / config.tick)
                profile["buy" if order.side == 1 else "sell"][distance] += order.quantity
    cash_error, share_error = math.fsum(cash) - initial_cash, math.fsum(shares) - initial_shares
    checks = {"cash_error": cash_error, "share_error": share_error,
              "minimum_cash": min(cash), "minimum_shares": min(shares),
              "reserved_cash_error": max((book.reserved_cash[i] - cash[i]) for i in range(config.agents)),
              "reserved_shares_error": max((book.reserved_shares[i] - shares[i]) for i in range(config.agents))}
    if abs(cash_error) > 1e-4 or abs(share_error) > 1e-6 or min(cash) < -1e-6 or min(shares) < -1e-8:
        raise ArithmeticError(f"资金库存检查失败：{checks}")
    if checks["reserved_cash_error"] > 1e-5 or checks["reserved_shares_error"] > 1e-7:
        raise ArithmeticError(f"预留约束检查失败：{checks}")
    metadata = {"config": asdict(config), "checks": checks, "variance_floor_events": floor_count,
                "budget_bindings": book.budget_bindings, "self_cancellations": book.self_cancellations,
                "horizon_min": min(horizons), "horizon_max": max(horizons), "horizon_mean": float(np.mean(horizons)),
                "profile_samples": profile_samples,
                "book_profile": {side: {str(k): v / max(1, profile_samples) for k, v in sorted(profile[side].items())}
                                 for side in profile}}
    return data, metadata
