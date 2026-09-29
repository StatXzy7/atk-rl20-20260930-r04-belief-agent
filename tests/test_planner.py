import unittest
from fractions import Fraction as F

from budgeted_agent import OptimalPlanner, Problem, Tool
from budgeted_agent import examples
from budgeted_agent.planner import STOP


class TestPlannerEdgeCases(unittest.TestCase):
    def test_no_budget_means_stop(self):
        p = examples.noisy_channel(budget=0)
        pl = OptimalPlanner(p)
        self.assertEqual(pl.value(), F(1, 2))
        self.assertIs(pl.policy(), STOP)

    def test_unaffordable_tool_is_skipped(self):
        p = examples.noisy_channel(budget=0)
        self.assertEqual(planner_value_helper(p), F(1, 2))

    def test_degenerate_prior_stops_immediately(self):
        chan = Tool("chan", 1, ((F(3, 4), F(1, 4)), (F(1, 4), F(3, 4))))
        for kappa in (F(0), F(1, 10)):
            p = Problem((F(1), F(0)), (chan,), kappa, 5)
            pl = OptimalPlanner(p)
            self.assertEqual(pl.value(), F(1))
            self.assertIs(pl.policy(), STOP)

    def test_deterministic_revealing_tool(self):
        det = Tool("det", 1, ((F(1), F(0)), (F(0), F(1))))
        p = Problem((F(1, 2), F(1, 2)), (det,), F(1, 4), 1)
        pl = OptimalPlanner(p)
        self.assertEqual(pl.value(), F(3, 4))  # 1 - kappa * 1
        self.assertEqual(pl.policy(), 0)

    def test_zero_probability_outcomes_are_skipped(self):
        # outcome 1 is impossible under every z; the tool is deterministic.
        tool = Tool("t", 1, ((F(1), F(0), F(0)), (F(1), F(0), F(0))))
        p = Problem((F(1, 2), F(1, 2)), (tool,), F(0), 1)
        pl = OptimalPlanner(p)
        self.assertEqual(pl.value(), F(1, 2))  # tool is uninformative

    def test_noisy_channel_exact_value(self):
        p = examples.noisy_channel()
        pl = OptimalPlanner(p)
        self.assertEqual(pl.value(), F(13, 20))
        self.assertEqual(pl.policy(), 0)
        # after one call the posterior is 3/4-confident; a second call is not
        # worth kappa: stop.
        self.assertIs(pl.policy((F(3, 4), F(1, 4)), 1), STOP)

    def test_tie_break_prefers_stop(self):
        # belief exactly at the indifference point 13/20.
        p = examples.rounding_trap_problem()
        pl = OptimalPlanner(p)
        b = (F(13, 20), F(7, 20))
        self.assertEqual(pl.value(b, 1), F(13, 20))
        self.assertIs(pl.policy(b, 1), STOP)

    def test_policy_is_deterministic_across_runs(self):
        p = examples.complementary_bits()
        a = OptimalPlanner(p).policy()
        b = OptimalPlanner(p).policy()
        self.assertEqual(a, b)

    def test_exact_cache_never_merges_close_beliefs(self):
        p = examples.rounding_trap_problem()
        pl = OptimalPlanner(p)
        pl.value(examples.ROUNDING_BELOW, 1)
        pl.value(examples.ROUNDING_ABOVE, 1)
        root_keys = [k for k in pl._cache if k[1] == 1]
        self.assertEqual(len(root_keys), 2)
        self.assertEqual(pl.stats()["cache_size"], 6)


def planner_value_helper(p):
    return OptimalPlanner(p).value()


if __name__ == "__main__":
    unittest.main()
