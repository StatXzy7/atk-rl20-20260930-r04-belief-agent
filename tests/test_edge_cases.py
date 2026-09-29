"""Edge cases: zero-probability observations, degenerate priors, exhausted
budget, deterministic tools, kappa = 0, and model validation."""
from fractions import Fraction as F

import pytest

from budgeted_info_agent import ExactPlanner, Problem, Tool, update
from budgeted_info_agent.belief import outcome_prob
from budgeted_info_agent.examples import bsc_example, parity_example


def _binary_problem(likelihood, prior=(F(1, 2), F(1, 2)), kappa=F(1, 10), budget=2):
    tool = Tool("t", 1, ("o0", "o1"), likelihood)
    return Problem(("z0", "z1"), prior, (tool,), kappa, budget)


def test_zero_probability_observation_update_returns_none():
    prob = _binary_problem(((F(1), F(0)), (F(0), F(1))))
    # observing o1 from a belief concentrated on z0 is impossible
    assert update((F(1), F(0)), prob.tools[0], 1) is None
    assert outcome_prob((F(1), F(0)), prob.tools[0], 1) == 0


def test_zero_probability_branch_pruned_in_planning():
    # deterministic tool: each branch has prob 0 or 1; planner must not
    # divide by zero and must reach value 1 - kappa exactly
    prob = _binary_problem(((F(1), F(0)), (F(0), F(1))))
    planner = ExactPlanner(prob)
    assert planner.value(prob.prior, prob.budget) == F(9, 10)
    assert planner.action(prob.prior, prob.budget) == ("call", 0)


def test_degenerate_prior_stops_immediately():
    prob = _binary_problem(
        ((F(1), F(0)), (F(0), F(1))), prior=(F(1), F(0))
    )
    planner = ExactPlanner(prob)
    assert planner.value(prob.prior, prob.budget) == 1
    assert planner.action(prob.prior, prob.budget) == ("stop", 0)


def test_degenerate_prior_kappa_zero():
    prob = _binary_problem(
        ((F(1), F(0)), (F(0), F(1))), prior=(F(1), F(0)), kappa=F(0)
    )
    planner = ExactPlanner(prob)
    assert planner.value(prob.prior, prob.budget) == 1
    assert planner.action(prob.prior, prob.budget)[0] == "stop"


def test_no_remaining_budget():
    prob = bsc_example(budget=0)
    planner = ExactPlanner(prob)
    assert planner.value(prob.prior, prob.budget) == F(1, 2)
    assert planner.action(prob.prior, prob.budget) == ("stop", 0)


def test_unaffordable_tool_never_called():
    # budget 1, tool cost 2: only stopping is feasible
    tool = Tool("expensive", 2, ("o0", "o1"),
                ((F(1), F(0)), (F(0), F(1))))
    prob = Problem(("z0", "z1"), (F(1, 2), F(1, 2)), (tool,), F(0), 1)
    planner = ExactPlanner(prob)
    assert planner.value(prob.prior, prob.budget) == F(1, 2)
    assert planner.action(prob.prior, prob.budget) == ("stop", 0)


def test_deterministic_tool_parity_example_uses_only_deterministic_calls():
    prob = parity_example()
    for tool in prob.tools:
        for row in tool.likelihood:
            assert set(row) <= {F(0), F(1)}
    planner = ExactPlanner(prob)
    assert planner.value(prob.prior, prob.budget) == F(2, 5)


def test_kappa_zero_values_are_pure_accuracy():
    prob = _binary_problem(
        ((F(3, 4), F(1, 4)), (F(1, 4), F(3, 4))), kappa=F(0), budget=1
    )
    planner = ExactPlanner(prob)
    assert planner.value(prob.prior, prob.budget) == F(3, 4)


def test_model_validation():
    with pytest.raises(ValueError):
        Tool("bad", 0, ("o0",), ((F(1),),))  # non-positive cost
    with pytest.raises(ValueError):
        Tool("bad", 1, ("o0", "o1"), ((F(1), F(1)),))  # row sums to 2
    with pytest.raises(ValueError):
        Problem(("z0",), (F(1, 2),), (), F(0), 0)  # prior does not sum to 1
    with pytest.raises(ValueError):
        Problem(("z0",), (F(1),), (), F(-1, 10), 0)  # negative kappa
    with pytest.raises(ValueError):
        Problem(("z0",), (F(1),), (), F(0), -1)  # negative budget
