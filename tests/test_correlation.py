"""Failure counterexample: when tool outputs are correlated given z, the
state (belief over z, remaining budget) is no longer sufficient."""
from fractions import Fraction as F

from budgeted_info_agent import ExactPlanner, outcome_prob, update
from budgeted_info_agent.examples import (
    correlation_example,
    correlated_world_value,
)


def test_planner_value_differs_from_correlated_world_value():
    env = correlation_example()
    prob = env["problem"]
    planner = ExactPlanner(prob)
    v_independent_model = planner.value(prob.prior, prob.budget)
    v_correlated = correlated_world_value(
        prob.prior, env["correlated_joint"], n_calls=3
    )
    assert v_independent_model == F(27, 32)
    assert v_correlated == F(3, 4)
    assert v_independent_model != v_correlated


def test_same_belief_state_different_predictive_distribution():
    env = correlation_example()
    prob = env["problem"]
    tool = prob.tools[0]
    # after observing o=0 once, the posterior over z is (3/4, 1/4) in BOTH
    # worlds (the per-tool likelihoods are identical)
    belief = update(prob.prior, tool, 0)
    assert belief == (F(3, 4), F(1, 4))
    # independence model predicts the next output:
    p_next_independent = outcome_prob(belief, tool, 0)
    assert p_next_independent == F(5, 8)
    # correlated truth: outputs coincide, so P(o2=0 | o1=0) = 1
    joint = env["correlated_joint"]
    p_o1_0 = sum(joint[(z, outs)] * prob.prior[z]
                 for z in (0, 1) for outs in joint_outcomes(joint, z)
                 if outs[0] == 0)
    p_both_0 = sum(joint[(z, outs)] * prob.prior[z]
                   for z in (0, 1) for outs in joint_outcomes(joint, z)
                   if outs[0] == 0 and outs[1] == 0)
    p_next_correlated = p_both_0 / p_o1_0
    assert p_next_correlated == 1
    # identical (belief, budget) state, different futures: the state is
    # insufficient once conditional independence fails
    assert p_next_independent != p_next_correlated


def joint_outcomes(joint, z):
    return [outs for (zz, outs) in joint if zz == z]
