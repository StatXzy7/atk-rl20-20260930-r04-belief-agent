import unittest
from fractions import Fraction as F

from budgeted_agent import Problem, Tool
from budgeted_agent.model import dumps, loads, problem_from_dict, problem_to_dict


def valid_tool():
    return Tool("t", 1, ((F(1, 2), F(1, 2)), (F(1, 2), F(1, 2))))


class TestModelValidation(unittest.TestCase):
    def test_valid_problem(self):
        Problem((F(1, 2), F(1, 2)), (valid_tool(),), F(0), 1)

    def test_prior_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            Problem((F(1, 2), F(1, 3)), (valid_tool(),), F(0), 1)

    def test_negative_prior_rejected(self):
        with self.assertRaises(ValueError):
            Problem((F(3, 2), F(-1, 2)), (valid_tool(),), F(0), 1)

    def test_cost_must_be_positive_integer(self):
        with self.assertRaises(ValueError):
            Problem((F(1),), (Tool("t", 0, ((F(1),),)),), F(0), 1)

    def test_likelihood_row_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            Problem((F(1),), (Tool("t", 1, ((F(1, 2), F(1, 3)),)),), F(0), 1)

    def test_negative_kappa_rejected(self):
        with self.assertRaises(ValueError):
            Problem((F(1),), (Tool("t", 1, ((F(1),),)),), F(-1), 1)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            Problem((F(1),), (Tool("t", 1, ((F(1),),)),), F(0), -1)

    def test_degenerate_prior_and_zero_likelihood_allowed(self):
        Problem((F(1), F(0)), (Tool("t", 1, ((F(1), F(0)), (F(0), F(1)))),), F(0), 1)

    def test_json_roundtrip(self):
        p = Problem(
            (F(1, 3), F(2, 3)),
            (Tool("a", 2, ((F(1, 7), F(6, 7)), (F(2, 3), F(1, 3)))),),
            F(1, 5),
            3,
        )
        q = loads(dumps(p))
        self.assertEqual(p, q)
        self.assertEqual(problem_to_dict(p), problem_to_dict(q))
        self.assertEqual(problem_from_dict(problem_to_dict(p)), p)


if __name__ == "__main__":
    unittest.main()
