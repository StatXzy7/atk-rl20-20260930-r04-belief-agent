"""Failure counterexamples, all with hand-checkable exact values."""
import unittest
from fractions import Fraction as F

from budgeted_agent import (
    OptimalPlanner,
    RoundedCachePlanner,
    fixed_order_value,
    greedy_policy,
    immediate_stop_value,
    policy_value,
)
from budgeted_agent import examples
from budgeted_agent.infogain import expected_information_gain
from budgeted_agent.planner import STOP


class TestGreedyFails(unittest.TestCase):
    """complementary_bits: optimal budget planning strictly beats one-step
    greedy. Hand computation (kappa=0, budget 2):
    stop              = 1/4
    pair_id (cost 2)  = 1/4*1 + 1/4*1 + 1/2*1/2 = 3/4
    parity / high_bit = 1/2 each (one bit known, MAP over 2 states)
    parity + high_bit = 1 (both bits known, z certain)
    Greedy compares one-step values 3/4 > 1/2, buys pair_id, ends at 3/4.
    Optimal buys both bits and reaches 1.
    """

    def test_exact_values_kappa_zero(self):
        p = examples.complementary_bits()
        self.assertEqual(immediate_stop_value(p), F(1, 4))
        self.assertEqual(fixed_order_value(p, [0]), F(3, 4))
        self.assertEqual(fixed_order_value(p, [1, 2]), F(1))
        self.assertEqual(policy_value(p, greedy_policy(p)), F(3, 4))
        self.assertEqual(OptimalPlanner(p).value(), F(1))

    def test_optimal_strictly_beats_greedy(self):
        p = examples.complementary_bits()
        self.assertGreater(OptimalPlanner(p).value(), policy_value(p, greedy_policy(p)))

    def test_strict_gap_survives_positive_kappa(self):
        p = examples.complementary_bits(F(1, 16))
        self.assertEqual(OptimalPlanner(p).value(), F(7, 8))
        self.assertEqual(policy_value(p, greedy_policy(p)), F(5, 8))

    def test_greedy_picks_the_wrong_tool(self):
        p = examples.complementary_bits()
        self.assertEqual(greedy_policy(p)(p.prior, p.budget), 0)
        self.assertEqual(OptimalPlanner(p).policy(), 1)


class TestEigTrap(unittest.TestCase):
    """Max information gain != max final decision value (budget 1, kappa 0)."""

    def test_eig_and_value_rank_tools_oppositely(self):
        p = examples.eig_trap()
        eig_tail = expected_information_gain(p, p.prior, 0)
        eig_leader = expected_information_gain(p, p.prior, 1)
        self.assertGreater(eig_tail, eig_leader)
        v_tail = policy_value(p, lambda b, B: 0 if B >= 1 else STOP)
        v_leader = policy_value(p, lambda b, B: 1 if B >= 1 else STOP)
        self.assertEqual(v_tail, F(1, 2))
        self.assertEqual(v_leader, F(23, 40))
        self.assertGreater(v_leader, v_tail)

    def test_planner_follows_value_not_eig(self):
        p = examples.eig_trap()
        pl = OptimalPlanner(p)
        self.assertEqual(pl.value(), F(23, 40))
        self.assertEqual(pl.policy(), 1)


class TestRoundingTrap(unittest.TestCase):
    """Float-approximate cache merging can flip the optimal action."""

    def test_exact_planner_separates_the_beliefs(self):
        p = examples.rounding_trap_problem()
        pl = OptimalPlanner(p)
        self.assertEqual(pl.policy(examples.ROUNDING_BELOW, 1), 0)
        self.assertIs(pl.policy(examples.ROUNDING_ABOVE, 1), STOP)
        self.assertEqual(pl.value(examples.ROUNDING_BELOW, 1), F(13, 20))
        self.assertEqual(pl.value(examples.ROUNDING_ABOVE, 1), F(651, 1000))

    def test_rounded_cache_merges_and_errs(self):
        p = examples.rounding_trap_problem()
        below_first = RoundedCachePlanner(p, ndigits=2)
        below_first.value(examples.ROUNDING_BELOW, 1)
        above_first = RoundedCachePlanner(p, ndigits=2)
        above_first.value(examples.ROUNDING_ABOVE, 1)
        self.assertEqual(
            below_first.policy(examples.ROUNDING_BELOW, 1),
            below_first.policy(examples.ROUNDING_ABOVE, 1),
            "rounded keys must collide within one cache",
        )
        self.assertEqual(
            above_first.policy(examples.ROUNDING_BELOW, 1),
            above_first.policy(examples.ROUNDING_ABOVE, 1),
            "rounded keys must collide within one cache",
        )
        correct = {examples.ROUNDING_BELOW: 0, examples.ROUNDING_ABOVE: STOP}
        for planner in (below_first, above_first):
            wrong = [
                b
                for b in (examples.ROUNDING_BELOW, examples.ROUNDING_ABOVE)
                if planner.policy(b, 1) != correct[b]
            ]
            self.assertTrue(wrong, "rounded cache must err on one of them")


class TestAdaptiveSavings(unittest.TestCase):
    def test_optimal_beats_fixed_order(self):
        p = examples.adaptive_savings()
        self.assertEqual(OptimalPlanner(p).value(), F(29, 32))
        self.assertEqual(fixed_order_value(p, [0, 1]), F(7, 8))
        self.assertGreater(F(29, 32), F(7, 8))


if __name__ == "__main__":
    unittest.main()
