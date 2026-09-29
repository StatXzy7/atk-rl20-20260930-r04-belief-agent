"""End-to-end demo: exact values, baselines, independent checks, sampling.

Run: python3 demo.py
"""
from fractions import Fraction as F

from budgeted_info_agent import (
    ExactPlanner,
    brute_force_optimal,
    fixed_order_value,
    greedy_policy,
    immediate_stop,
    policy_value,
    simulate,
    unrolled_optimal_value,
)
from budgeted_info_agent.examples import (
    bsc_example,
    correlation_example,
    correlated_world_value,
    infogain_trap_example,
    parity_example,
)
from budgeted_info_agent.infogain import (
    expected_decision_value,
    expected_info_gain,
)


def main() -> None:
    print("== parity example: optimal strictly beats one-step greedy ==")
    prob = parity_example()
    planner = ExactPlanner(prob)
    optimal = planner.value(prob.prior, prob.budget)
    stop_v, _ = immediate_stop(prob)
    greedy_v = policy_value(prob, greedy_policy(prob))
    print(f"  immediate stop      : {stop_v}  = {float(stop_v):.4f}")
    print(f"  one-step greedy     : {greedy_v}  = {float(greedy_v):.4f}")
    print(f"  fixed order (1 call): {fixed_order_value(prob, (0,))}")
    print(f"  fixed order (2 call): {fixed_order_value(prob, (0, 1))}")
    print(f"  OPTIMAL             : {optimal}  = {float(optimal):.4f}")
    best, n = brute_force_optimal(prob)
    print(f"  brute-force check over {n} policies: {best}")
    counter = [0]
    print(f"  unrolled tree check : {unrolled_optimal_value(prob, counter=counter)}"
          f"  ({counter[0]} nodes)")

    print("\n== bsc example: seeded sampling vs exact value ==")
    prob = bsc_example(budget=2)
    planner = ExactPlanner(prob)
    exact = planner.value(prob.prior, prob.budget)
    result = simulate(prob, planner.policy, n_episodes=4000, seed=20260930)
    print(f"  exact V = {exact} = {float(exact):.4f}")
    print(f"  sampled mean = {float(result['mean']):.4f}"
          f"  (stderr {result['stderr']:.4f}, seed=20260930)")

    print("\n== info gain != decision value ==")
    prob = infogain_trap_example()
    for i, name in enumerate(("A: resolve-tail", "B: separate-z0")):
        ig = expected_info_gain(prob, prob.prior, i)
        dv = expected_decision_value(prob, prob.prior, i)
        print(f"  {name}: info gain {ig:.4f} bit, decision value {dv}")
    planner = ExactPlanner(prob)
    print(f"  planner picks tool index {planner.action(prob.prior, prob.budget)[1]}"
          f" (decision-optimal)")

    print("\n== tool correlation breaks the state definition ==")
    env = correlation_example()
    prob = env["problem"]
    v_ind = ExactPlanner(prob).value(prob.prior, prob.budget)
    v_cor = correlated_world_value(prob.prior, env["correlated_joint"], 3)
    print(f"  independent world (planner's model): V = {v_ind}")
    print(f"  perfectly correlated world (truth) : V = {v_cor}")
    print("  same per-tool likelihoods, same reachable beliefs, different values")


if __name__ == "__main__":
    main()
