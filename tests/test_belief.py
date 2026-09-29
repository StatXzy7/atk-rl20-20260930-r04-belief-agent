import unittest
from fractions import Fraction as F

from budgeted_agent import Tool
from budgeted_agent.belief import best_guess, marginal, stop_value, update


class TestBelief(unittest.TestCase):
    def setUp(self):
        self.chan = Tool("chan", 1, ((F(3, 4), F(1, 4)), (F(1, 4), F(3, 4))))

    def test_update_matches_hand_computation(self):
        post = update((F(1, 2), F(1, 2)), self.chan, 0)
        self.assertEqual(post, (F(3, 4), F(1, 4)))
        post2 = update(post, self.chan, 1)
        self.assertEqual(post2, (F(1, 2), F(1, 2)))

    def test_updates_commute(self):
        b = (F(1, 2), F(1, 2))
        ab = update(update(b, self.chan, 0), self.chan, 1)
        ba = update(update(b, self.chan, 1), self.chan, 0)
        self.assertEqual(ab, ba)

    def test_zero_probability_observation_rejected(self):
        det = Tool("det", 1, ((F(1), F(0)), (F(0), F(1))))
        with self.assertRaises(ValueError):
            update((F(1), F(0)), det, 1)

    def test_degenerate_prior_stays_degenerate(self):
        post = update((F(1), F(0)), self.chan, 0)
        self.assertEqual(post, (F(1), F(0)))

    def test_deterministic_tool_collapses_belief(self):
        det = Tool("det", 1, ((F(1), F(0)), (F(0), F(1))))
        self.assertEqual(update((F(1, 2), F(1, 2)), det, 0), (F(1), F(0)))
        self.assertEqual(update((F(1, 2), F(1, 2)), det, 1), (F(0), F(1)))

    def test_marginal_sums_to_one_over_outcomes(self):
        b = (F(2, 5), F(3, 5))
        self.assertEqual(marginal(b, self.chan, 0) + marginal(b, self.chan, 1), 1)

    def test_stop_value_and_guess(self):
        self.assertEqual(stop_value((F(1, 3), F(2, 3))), F(2, 3))
        self.assertEqual(best_guess((F(1, 3), F(2, 3))), 1)
        self.assertEqual(best_guess((F(1, 2), F(1, 2))), 0)  # deterministic tie-break


if __name__ == "__main__":
    unittest.main()
