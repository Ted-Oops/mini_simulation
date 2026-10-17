import math
import random
import unittest
import numpy as np
from market.order_book import OrderBook
from market.config import MarketConfig
from market.engine import price_bounds, simulate
from market.statistics import acf, hill


class MarketTests(unittest.TestCase):
    def test_roots_satisfy_portfolio_and_budget_equations(self):
        for av in (1e-12, 1e-7, 1e-3):
            for shares in (0, 25, 100):
                for cash in (0, 7500, 15000):
                    low, star, high = price_bounds(math.log(300), av, shares, cash)
                    self.assertLessEqual(low, star + 1e-9)
                    self.assertLessEqual(star, high + 1e-9)
                    if low > 1e-250:
                        self.assertAlmostEqual(math.log(300 / low), av * (shares * low + cash), delta=1e-10)
                    self.assertAlmostEqual(math.log(300 / star), av * shares * star, delta=1e-10)

    def test_price_priority_partial_fills_and_budget_reservation(self):
        cash, shares = [1000., 0., 0.], [0., 5., 5.]
        book = OrderBook(cash, shares, 1.)
        book.submit(1, -1, 100, 5, 1, 20)
        book.submit(2, -1, 101, 5, 2, 20)
        last, volume, rest = book.submit(0, 1, 101, 10, 3, 20)
        self.assertEqual(last, 101)
        self.assertAlmostEqual(volume, 5 + 500 / 101)
        self.assertEqual(rest, 0)
        self.assertAlmostEqual(sum(cash), 1000)
        self.assertAlmostEqual(sum(shares), 10)
        self.assertGreaterEqual(min(cash), -1e-10)
        self.assertGreaterEqual(min(shares), -1e-10)

    def test_multiple_orders_reserve_and_expire_independently(self):
        book = OrderBook([1000., 0.], [10., 10.], 1.)
        book.submit(0, 1, 10, 30, 1, 3)
        book.submit(0, 1, 11, 20, 2, 6)
        self.assertEqual(len(book.orders), 2)
        self.assertEqual(book.available_cash(0), 480)
        book.expire(4)
        self.assertEqual(len(book.orders), 1)
        self.assertEqual(book.available_cash(0), 780)
        book.cancel_owner(0)
        self.assertEqual(book.available_cash(0), 1000)

    def test_next_price_gap_uses_distinct_price_levels(self):
        book = OrderBook([1000.] * 3, [10.] * 3, 1.)
        book.submit(0, 1, 10, 2, 1, 100)
        book.submit(1, 1, 10, 2, 2, 100)
        book.submit(2, 1, 8, 2, 3, 100)
        self.assertEqual(book.gap(1), 2)
        self.assertEqual(book.best(1), 10)

    def test_original_horizon_is_not_capped_and_paths_replay(self):
        config = MarketConfig(agents=40, steps=300, seed=2, burn_in=20, profile_stride=10)
        first, metadata = simulate(config)
        second, again = simulate(config)
        self.assertGreater(metadata['horizon_max'], 200)
        for name in first:
            np.testing.assert_equal(first[name], second[name])
        self.assertEqual(metadata, again)
        self.assertAlmostEqual(metadata['checks']['cash_error'], 0, places=6)
        self.assertAlmostEqual(metadata['checks']['share_error'], 0, places=8)

    def test_fft_autocorrelation_matches_direct_estimator(self):
        values = np.random.default_rng(8).normal(size=2000)
        centered = values - values.mean()
        direct = np.dot(centered[17:], centered[:-17]) / np.dot(centered, centered)
        self.assertAlmostEqual(acf(values, 20)[16], direct)
        short = acf(values[:3], 20)
        self.assertEqual(len(short), 20)
        self.assertTrue(all(value is None for value in short[2:]))
        self.assertIsNone(hill(np.ones(1000)))

    def test_heap_book_matches_independent_sorted_list_ledger(self):
        # 对照账本直接重算每笔预留，无价格堆或预留缓存，检查优化撮合的实际成交。
        rng = random.Random(27)
        cash, shares = [1000.] * 5, [30.] * 5
        oracle_cash, oracle_shares = cash.copy(), shares.copy()
        book = OrderBook(cash, shares, 1.)
        orders = {}
        sequence = 0

        def available(owner, buy):
            reserved = sum(o['quantity'] * (o['ticks'] if buy else 1) for o in orders.values()
                           if o['owner'] == owner and o['side'] == (1 if buy else -1))
            return max(0., (oracle_cash if buy else oracle_shares)[owner] - reserved)

        for step in range(1, 501):
            book.expire(step)
            orders = {k:o for k,o in orders.items() if o['expiry'] > step}
            owner, side, ticks = rng.randrange(5), rng.choice([-1,1]), rng.randrange(10,41)
            quantity = rng.uniform(.1,60)
            remaining = quantity if side == 1 else min(quantity,available(owner,False))
            volume, last = 0., None
            while remaining > 1e-10:
                opposite = sorted([(key,o) for key,o in orders.items() if o['side'] == -side],
                                  key=lambda item:(item[1]['ticks'] * side,item[0]))
                if not opposite:
                    break
                key, resting = opposite[0]
                price = resting['ticks']
                if (side == 1 and price > ticks) or (side == -1 and price < ticks):
                    break
                if resting['owner'] == owner:
                    del orders[key]
                    continue
                feasible = available(owner,True) / price if side == 1 else available(owner,False)
                amount = min(remaining,resting['quantity'],feasible)
                if amount <= 1e-10:
                    break
                buyer,seller = (owner,resting['owner']) if side == 1 else (resting['owner'],owner)
                oracle_cash[buyer] -= amount * price; oracle_cash[seller] += amount * price
                oracle_shares[buyer] += amount; oracle_shares[seller] -= amount
                remaining -= amount; resting['quantity'] -= amount; volume += amount; last = float(price)
                if resting['quantity'] <= 1e-10:
                    del orders[key]
            rest = min(remaining,available(owner,True)/ticks if side == 1 else available(owner,False))
            lifetime = rng.randrange(1,40)
            if rest > 1e-10:
                sequence += 1
                orders[sequence] = dict(owner=owner,side=side,ticks=ticks,quantity=rest,expiry=step+lifetime)
            actual_last,actual_volume,actual_rest = book.submit(owner,side,ticks,quantity,step,lifetime)
            self.assertEqual(actual_last,last)
            self.assertAlmostEqual(actual_volume,volume,places=7)
            self.assertAlmostEqual(actual_rest,rest,places=7)
            np.testing.assert_allclose(cash,oracle_cash,rtol=0,atol=1e-7)
            np.testing.assert_allclose(shares,oracle_shares,rtol=0,atol=1e-8)
            for direction in [-1,1]:
                expected = [o['ticks'] for o in orders.values() if o['side'] == direction]
                best = (max(expected) if direction == 1 else min(expected)) if expected else None
                self.assertEqual(book.best(direction),best)


if __name__ == '__main__':
    unittest.main()
