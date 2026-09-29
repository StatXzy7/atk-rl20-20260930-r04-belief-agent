"""Independent verification: brute-force policy enumeration and explicit
path sums must agree with the Bellman planner."""
import unittest
from fractions import Fraction as F

from budgeted_agent import OptimalPlanner, examples, greedy_policy, policy_value
from budgeted_agent.enumeration import (
    count_tree_paths,
    enumerate_all_policies,
    policy_value_by_paths,
    reachable_states,
)


class TestBruteForceOptimality(unittest.TestCase):
    def test_noisy_channel_all_policies(self):
        p = examples.noisy_channel()
        best, assign, n = enumerate_all_policies(p)
        self.assertEqual(n, 8)
        self.assertEqual(best, F(13, 20))
        self.assertEqual(OptimalPlanner(p).value(), best)

    def test_complementary_bits_all_policies(self):
        p = examples.complementary_bits()
        best, assign, n = enumerate_all_policies(p)
        self.assertEqual(n, 324)
        self.assertEqual(best, F(1))
        self.assertEqual(OptimalPlanner(p).value(), best)

    def test_planner_policy_matches_enumeration_value_by_paths(self):
        problems = (
            examples.noisy_channel(),
            examples.complementary_bits(),
            examples.adaptive_savings(),
            examples.eig_trap(),
        )
        for p in problems:
            pl = OptimalPlanner(p)
            by_paths, n_paths = policy_value_by_paths(p, pl.policy)
            self.assertEqual(by_paths, pl.value())
            self.assertGreater(n_paths, 0)

    def test_greedy_value_by_paths_matches_recursion(self):
        p = examples.complementary_bits()
        by_paths, _ = policy_value_by_paths(p, greedy_policy(p))
        self.assertEqual(by_paths, F(3, 4))
        self.assertEqual(by_paths, policy_value(p, greedy_policy(p)))


class TestStateSpace(unittest.TestCase):
    def test_cache_is_much_smaller_than_path_count(self):
        p = examples.noisy_channel(budget=4)
        pl = OptimalPlanner(p)
        pl.value()
        cache_size = pl.stats()["cache_size"]
        n_paths = count_tree_paths(p)
        self.assertLess(cache_size, n_paths)
        self.assertEqual(cache_size, len(reachable_states(p)))

    def test_reachable_states_respect_budget(self):
        p = examples.complementary_bits()
        for b, B in reachable_states(p):
            self.assertGreaterEqual(B, 0)
            self.assertLessEqual(B, p.budget)
            self.assertEqual(sum(b), 1)


if __name__ == "__main__":
    unittest.main()
