"""Independent verification: full tree unrolling and brute-force enumeration
over all adaptive policies must agree with the memoized planner."""
from fractions import Fraction as F

from budgeted_info_agent import (
    ExactPlanner,
    brute_force_optimal,
    policy_value,
    unrolled_optimal_value,
)
from budgeted_info_agent.examples import bsc_example, parity_example


def test_parity_unrolled_matches_planner():
    prob = parity_example()
    planner = ExactPlanner(prob)
    memoized = planner.value(prob.prior, prob.budget)
    counter = [0]
    unrolled = unrolled_optimal_value(prob, counter=counter)
    assert unrolled == memoized == F(2, 5)
    assert counter[0] == 17  # explicit tree nodes (zero-prob branches pruned)


def test_parity_brute_force_all_policies():
    prob = parity_example()
    best, n_policies = brute_force_optimal(prob)
    assert n_policies == 243  # 5 decision nodes, 3 actions each
    assert best == F(2, 5)
    # the planner's own policy attains the brute-force optimum
    planner = ExactPlanner(prob)
    assert policy_value(prob, planner.policy()) == best


def test_bsc_unrolled_and_brute_force_match_planner():
    prob = bsc_example(budget=2)
    planner = ExactPlanner(prob)
    memoized = planner.value(prob.prior, prob.budget)
    assert unrolled_optimal_value(prob) == memoized
    best, n_policies = brute_force_optimal(prob)
    assert n_policies == 8  # 3 decision nodes, 2 actions each
    assert best == memoized == F(13, 20)


def test_state_space_growth_cache_vs_tree():
    # one binary tool, budget 6: the observation tree has 2^7 - 1 = 127
    # nodes, but only sum_{d=0}^{6} (d+1) = 28 distinct (belief, budget)
    # states, because the posterior depends only on the observation counts
    prob = bsc_example(budget=6)
    planner = ExactPlanner(prob)
    planner.value(prob.prior, prob.budget)
    assert len(planner.cache) == 28
    counter = [0]
    unrolled_optimal_value(prob, counter=counter)
    assert counter[0] == 127
