"""Optimal planner: exact values, induced policy, and the strict gap over
one-step greedy on the parity example."""
from fractions import Fraction as F

from budgeted_info_agent import ExactPlanner, greedy_policy, policy_value
from budgeted_info_agent.belief import update
from budgeted_info_agent.examples import bsc_example, parity_example


def test_parity_optimal_value_and_policy():
    prob = parity_example()
    planner = ExactPlanner(prob)
    assert planner.value(prob.prior, prob.budget) == F(2, 5)
    # root: call bit1 (or bit2; cache order makes bit1 canonical here)
    assert planner.action(prob.prior, prob.budget) == ("call", 0)
    # after observing bit1=0: belief (1/2, 1/2, 0, 0), call bit2
    b1 = update(prob.prior, prob.tools[0], 0)
    assert b1 == (F(1, 2), F(1, 2), F(0), F(0))
    assert planner.action(b1, 3) == ("call", 1)
    # after observing bit2=1: z01 certain, stop and report state 1
    b2 = update(b1, prob.tools[1], 1)
    assert b2 == (F(0), F(1), F(0), F(0))
    assert planner.action(b2, 0) == ("stop", 1)


def test_parity_optimal_strictly_beats_greedy():
    prob = parity_example()
    optimal = ExactPlanner(prob).value(prob.prior, prob.budget)
    greedy = policy_value(prob, greedy_policy(prob))
    assert greedy == F(1, 4)
    assert optimal > greedy  # 2/5 > 1/4, strict


def test_bsc_value_and_policy():
    prob = bsc_example(budget=2)
    planner = ExactPlanner(prob)
    assert planner.value(prob.prior, prob.budget) == F(13, 20)
    assert planner.action(prob.prior, prob.budget) == ("call", 0)
    # after one observation the posterior is 3/4 on some state: stop
    b1 = update(prob.prior, prob.tools[0], 0)
    assert b1 == (F(3, 4), F(1, 4))
    assert planner.action(b1, 1) == ("stop", 0)


def test_bsc_budget_variants():
    assert ExactPlanner(bsc_example(1)).value(*_prior_budget(bsc_example(1))) == F(13, 20)
    assert ExactPlanner(bsc_example(0)).value(*_prior_budget(bsc_example(0))) == F(1, 2)


def _prior_budget(prob):
    return prob.prior, prob.budget


def test_cache_uses_exact_beliefs_not_rounded():
    prob = bsc_example(2)
    planner = ExactPlanner(prob)
    planner.value(prob.prior, prob.budget)
    keys = list(planner.cache)
    # every cached belief is an exact rational tuple summing to 1
    for belief, budget in keys:
        assert sum(belief) == 1
        assert all(isinstance(x, F) for x in belief)
