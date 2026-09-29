"""Failure counterexample: maximizing expected information gain is not
maximizing expected final-decision value."""
from fractions import Fraction as F

from budgeted_info_agent import ExactPlanner
from budgeted_info_agent.examples import infogain_trap_example
from budgeted_info_agent.infogain import (
    expected_decision_value,
    expected_info_gain,
)


def test_infogain_and_decision_value_rank_tools_oppositely():
    prob = infogain_trap_example()
    ig_a = expected_info_gain(prob, prob.prior, 0)  # resolve-tail
    ig_b = expected_info_gain(prob, prob.prior, 1)  # separate-z0
    dv_a = expected_decision_value(prob, prob.prior, 0)
    dv_b = expected_decision_value(prob, prob.prior, 1)
    assert ig_a > ig_b            # entropy heuristic prefers A
    assert dv_a == F(13, 20)      # exact: 0.65
    assert dv_b == F(4, 5)        # exact: 0.80
    assert dv_a < dv_b            # decision value prefers B


def test_planner_picks_decision_optimal_tool():
    prob = infogain_trap_example()
    planner = ExactPlanner(prob)
    assert planner.action(prob.prior, prob.budget) == ("call", 1)
    assert planner.value(prob.prior, prob.budget) == F(4, 5)
