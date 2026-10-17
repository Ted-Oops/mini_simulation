"""连续数量、现金库存预留、价格时间优先和逐笔到期的订单簿。"""

import heapq
from dataclasses import dataclass


@dataclass(slots=True)
class Order:
    owner: int
    side: int
    ticks: int
    quantity: float
    expiry: int


class OrderBook:
    def __init__(self, cash, shares, tick):
        self.cash, self.shares, self.tick = cash, shares, tick
        self.reserved_cash = [0.0] * len(cash)
        self.reserved_shares = [0.0] * len(cash)
        self.orders = {}
        self.owner_orders = [set() for _ in cash]
        self.bids, self.asks, self.expiries = [], [], []
        self.sequence = 0
        self.self_cancellations = 0
        self.budget_bindings = 0

    def available_cash(self, owner):
        return max(0.0, self.cash[owner] - self.reserved_cash[owner])

    def available_shares(self, owner):
        return max(0.0, self.shares[owner] - self.reserved_shares[owner])

    def _release(self, order, quantity):
        if order.side == 1:
            self.reserved_cash[order.owner] = max(0.0, self.reserved_cash[order.owner] - quantity * order.ticks * self.tick)
        else:
            self.reserved_shares[order.owner] = max(0.0, self.reserved_shares[order.owner] - quantity)

    def cancel(self, sequence):
        order = self.orders.pop(sequence, None)
        if order is not None:
            self._release(order, order.quantity)
            self.owner_orders[order.owner].discard(sequence)

    def cancel_owner(self, owner):
        for sequence in tuple(self.owner_orders[owner]):
            self.cancel(sequence)

    def expire(self, step):
        while self.expiries and self.expiries[0][0] <= step:
            _, sequence = heapq.heappop(self.expiries)
            self.cancel(sequence)

    def best(self, side):
        heap = self.bids if side == 1 else self.asks
        while heap and heap[0][1] not in self.orders:
            heapq.heappop(heap)
        if not heap:
            return None
        return self.orders[heap[0][1]].ticks * self.tick

    def gap(self, side):
        heap = self.bids if side == 1 else self.asks
        first = self.best(side)
        if first is None:
            return None
        # 堆的第二项不一定是下一价格档；临时弹出整个最优价位再恢复。
        removed = []
        while heap:
            price, sequence = heap[0]
            if sequence not in self.orders:
                heapq.heappop(heap)
            elif self.orders[sequence].ticks * self.tick == first:
                removed.append(heapq.heappop(heap))
            else:
                break
        second = self.best(side)
        for item in removed:
            heapq.heappush(heap, item)
        return abs(second - first) if second is not None else None

    def submit(self, owner, side, ticks, quantity, step, lifetime, integer=False):
        """返回最后成交价、成交总量、剩余挂单量；主动成交按对手旧单价格。"""
        limit = ticks * self.tick
        quantity = max(0.0, quantity)
        if side == -1:
            quantity = min(quantity, self.available_shares(owner))
        if integer:
            quantity = float(int(quantity))
        remaining, volume, last_price = quantity, 0.0, None
        opposite = self.asks if side == 1 else self.bids
        while remaining > 1e-10:
            best = self.best(-side)
            if best is None or (side == 1 and best > limit) or (side == -1 and best < limit):
                break
            sequence = opposite[0][1]
            resting = self.orders[sequence]
            if resting.owner == owner:
                self.cancel(sequence)
                self.self_cancellations += 1
                continue
            feasible = self.available_cash(owner) / best if side == 1 else self.available_shares(owner)
            amount = min(remaining, resting.quantity, feasible)
            if integer:
                amount = float(int(amount + 1e-10))
            if amount <= 1e-10:
                self.budget_bindings += 1
                break
            buyer, seller = (owner, resting.owner) if side == 1 else (resting.owner, owner)
            self._release(resting, amount)
            self.cash[buyer] -= amount * best
            self.cash[seller] += amount * best
            self.shares[buyer] += amount
            self.shares[seller] -= amount
            remaining -= amount
            resting.quantity -= amount
            volume += amount
            last_price = best
            if resting.quantity <= 1e-10:
                self.cancel(sequence)
        feasible = self.available_cash(owner) / limit if side == 1 else self.available_shares(owner)
        rest = min(remaining, feasible)
        if rest + 1e-10 < remaining:
            self.budget_bindings += 1
        if integer:
            rest = float(int(rest + 1e-10))
        if rest > 1e-10:
            self.sequence += 1
            sequence = self.sequence
            self.orders[sequence] = Order(owner, side, ticks, rest, step + lifetime)
            self.owner_orders[owner].add(sequence)
            heapq.heappush(self.bids if side == 1 else self.asks, (-ticks if side == 1 else ticks, sequence))
            heapq.heappush(self.expiries, (step + lifetime, sequence))
            if side == 1:
                self.reserved_cash[owner] += rest * limit
            else:
                self.reserved_shares[owner] += rest
        return last_price, volume, rest
