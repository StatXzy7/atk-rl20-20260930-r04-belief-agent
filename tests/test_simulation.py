"""Seeded simulation cross-checked against exact values; episode records
must be fully recomputable from the problem inputs."""
from fractions import Fraction as F

from budgeted_info_agent import ExactPlanner, replay_episode, simulate
from budgeted_info_agent.examples import bsc_example, parity_example


def test_simulation_matches_exact_value_parity_deterministic():
    prob = parity_example()
    planner = ExactPlanner(prob)
    planner.value(prob.prior, prob.budget)
    result = simulate(prob, planner.policy, n_episodes=200, seed=12345)
    # deterministic tools: every episode yields exactly 2/5, zero variance
    assert result["mean"] == F(2, 5)
    assert result["stderr"] < 1e-12  # ~0, up to float conversion of 2/5


def test_simulation_matches_exact_value_bsc_stochastic():
    prob = bsc_example(budget=2)
    planner = ExactPlanner(prob)
    exact = planner.value(prob.prior, prob.budget)
    result = simulate(prob, planner.policy, n_episodes=4000, seed=20260930)
    # exact value lies within 4 standard errors of the sampled mean
    assert abs(float(result["mean"] - exact)) <= 4 * result["stderr"]
    assert result["stderr"] < 0.02


def test_simulation_reproducible_under_fixed_seed():
    prob = bsc_example(budget=2)
    planner = ExactPlanner(prob)
    run1 = simulate(prob, planner.policy, n_episodes=100, seed=7)
    run2 = simulate(prob, planner.policy, n_episodes=100, seed=7)
    assert run1["mean"] == run2["mean"]
    assert [r.reward for r in run1["records"]] == [
        r.reward for r in run2["records"]
    ]


def test_episode_records_fully_recomputable():
    prob = bsc_example(budget=2)
    planner = ExactPlanner(prob)
    result = simulate(prob, planner.policy, n_episodes=50, seed=99)
    for record in result["records"]:
        assert replay_episode(prob, record) == record.reward


def test_policy_never_receives_hidden_state():
    prob = parity_example()
    planner = ExactPlanner(prob)
    seen_arities = []

    def spy_policy(belief, budget):  # exactly two arguments, no z
        seen_arities.append(2)
        return planner.action(belief, budget)

    simulate(prob, lambda: spy_policy, n_episodes=20, seed=1)
    assert seen_arities and all(n == 2 for n in seen_arities)
