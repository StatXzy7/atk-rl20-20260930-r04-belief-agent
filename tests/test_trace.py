"""Episode records must be fully recomputable from the logged inputs."""
import json
import random
import unittest
from fractions import Fraction as F

from budgeted_agent import OptimalPlanner, examples
from budgeted_agent.trace import record_episode, replay


class TestTrace(unittest.TestCase):
    def test_replay_recomputes_beliefs_costs_payoff(self):
        p = examples.noisy_channel()
        pl = OptimalPlanner(p)
        pl.value()
        rng = random.Random(123)
        rec = record_episode(p, pl.policy, z=0, rng=rng)
        out = replay(rec)
        self.assertEqual(out["spent"], 1)
        self.assertEqual(out["final_belief"], (F(3, 4), F(1, 4)))
        self.assertEqual(out["guess"], 0)
        self.assertTrue(out["correct"])
        self.assertEqual(out["payoff"], F(9, 10))

    def test_record_survives_json_roundtrip(self):
        p = examples.complementary_bits()
        pl = OptimalPlanner(p)
        pl.value()
        rng = random.Random(5)
        rec = record_episode(p, pl.policy, z=3, rng=rng)
        rec2 = json.loads(json.dumps(rec))
        out = replay(rec2)
        self.assertEqual(out["final_belief"], (F(0), F(0), F(0), F(1)))
        self.assertEqual(out["payoff"], F(1))

    def test_replay_rejects_budget_violation(self):
        p = examples.noisy_channel()
        pl = OptimalPlanner(p)
        pl.value()
        rng = random.Random(1)
        rec = record_episode(p, pl.policy, z=0, rng=rng)
        rec["steps"].append({"tool": 0, "observation": 0})
        rec["steps"].append({"tool": 0, "observation": 0})
        rec["steps"].append({"tool": 0, "observation": 0})
        with self.assertRaises(ValueError):
            replay(rec)

    def test_correlated_calls_break_state_sufficiency(self):
        demo = examples.correlation_demo()
        self.assertEqual(demo["belief_after_o1_eq_0"], demo["belief_after_o1_eq_1"])
        self.assertNotEqual(
            demo["P_o2_eq_0_given_o1_eq_0"], demo["P_o2_eq_0_given_o1_eq_1"]
        )


if __name__ == "__main__":
    unittest.main()
