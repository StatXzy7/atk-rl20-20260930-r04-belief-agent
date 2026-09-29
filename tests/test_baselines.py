"""The three baselines on the parity example, with hand-checkable numbers."""
from fractions import Fraction as F

from budgeted_info_agent import (
    ExactPlanner,
    FixedOrderPolicy,
    fixed_order_value,
    greedy_action,
    greedy_policy,
    immediate_stop,
    policy_value,
)
from budgeted_info_agent.examples import parity_example


def test_immediate_stop():
    prob = parity_example()
    value, action = immediate_stop(prob)
    assert value == F(1, 4)
    assert action == ("stop", 0)  # tie broken to smallest index


def test_greedy_stops_immediately_and_is_suboptimal():
    prob = parity_example()
    # one-step lookahead: call => 1/2 - 3/10 = 1/5 < 1/4, so greedy stops
    assert greedy_action(prob, prob.prior, prob.budget) == ("stop", 0)
    assert policy_value(prob, greedy_policy(prob)) == F(1, 4)


def test_fixed_order_values():
    prob = parity_example()
    assert fixed_order_value(prob, (0, 1)) == F(2, 5)  # both bits: 1 - 6/10
    assert fixed_order_value(prob, (1, 0)) == F(2, 5)
    assert fixed_order_value(prob, (0,)) == F(1, 5)    # one bit: 1/2 - 3/10
    assert fixed_order_value(prob, ()) == F(1, 4)      # no calls at all


def test_fixed_order_policy_matches_exact_value():
    prob = parity_example()
    policy = FixedOrderPolicy(prob, (0, 1)).reset()
    # stateful policy: evaluate by simulation-free path rollout
    from budgeted_info_agent.simulate import run_episode
    import random

    rng = random.Random(7)
    rewards = [
        run_episode(prob, FixedOrderPolicy(prob, (0, 1)).reset(), rng).reward
        for _ in range(50)
    ]
    assert all(r == F(2, 5) for r in rewards)  # deterministic tools


def test_baseline_ordering_on_parity():
    prob = parity_example()
    optimal = ExactPlanner(prob).value(prob.prior, prob.budget)
    stop_v, _ = immediate_stop(prob)
    greedy_v = policy_value(prob, greedy_policy(prob))
    fixed_full = fixed_order_value(prob, (0, 1))
    fixed_partial = fixed_order_value(prob, (0,))
    assert stop_v == greedy_v == F(1, 4)
    assert fixed_partial == F(1, 5) < stop_v
    assert fixed_full == optimal == F(2, 5)
    assert optimal > greedy_v
