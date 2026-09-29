"""Fixed-seed Monte Carlo must agree with the exact values."""
import unittest
from fractions import Fraction as F

from budgeted_agent import (
    FixedOrderPolicy,
    OptimalPlanner,
    examples,
    greedy_policy,
)
from budgeted_agent.simulation import simulate

SEED = 20260930
EPISODES = 4000
TOL_SIGMAS = 5.0


def check_close(test, sim, exact):
    exact_f = float(exact)
    tol = max(TOL_SIGMAS * sim.std_err, 1e-9)
    test.assertAlmostEqual(sim.mean, exact_f, delta=tol)


class TestMonteCarloVsExact(unittest.TestCase):
    def test_optimal_policy_noisy_channel(self):
        p = examples.noisy_channel()
        pl = OptimalPlanner(p)
        pl.value()
        sim = simulate(p, lambda: pl.policy, EPISODES, SEED)
        check_close(self, sim, F(13, 20))

    def test_greedy_policy_complementary_bits(self):
        p = examples.complementary_bits()
        sim = simulate(p, lambda: greedy_policy(p), EPISODES, SEED)
        check_close(self, sim, F(3, 4))

    def test_fixed_order_complementary_bits(self):
        p = examples.complementary_bits()
        sim = simulate(
            p, lambda: FixedOrderPolicy(p, [0]), EPISODES, SEED
        )
        check_close(self, sim, F(3, 4))

    def test_deterministic_seed_reproducible(self):
        p = examples.noisy_channel()
        pl = OptimalPlanner(p)
        pl.value()
        a = simulate(p, lambda: pl.policy, 500, 7)
        b = simulate(p, lambda: pl.policy, 500, 7)
        self.assertEqual(a.mean, b.mean)

    def test_optimal_complementary_bits_always_correct(self):
        # kappa=0 and the optimal plan identifies z with certainty.
        p = examples.complementary_bits()
        pl = OptimalPlanner(p)
        pl.value()
        sim = simulate(p, lambda: pl.policy, 500, SEED)
        self.assertEqual(sim.mean, 1.0)


if __name__ == "__main__":
    unittest.main()
