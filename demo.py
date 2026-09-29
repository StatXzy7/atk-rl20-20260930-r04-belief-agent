"""End-to-end demo: exact expected payoffs (Fractions), independent
enumeration checks, fixed-seed Monte Carlo vs exact, cache statistics,
and a fully replayable episode record. Run: python3 demo.py"""
import json
import random
from fractions import Fraction as F

from budgeted_agent import (
    FixedOrderPolicy,
    OptimalPlanner,
    RoundedCachePlanner,
    examples,
    fixed_order_value,
    greedy_policy,
    immediate_stop_value,
    policy_value,
)
from budgeted_agent.enumeration import (
    count_tree_paths,
    enumerate_all_policies,
    policy_value_by_paths,
    reachable_states,
)
from budgeted_agent.infogain import expected_information_gain
from budgeted_agent.simulation import simulate
from budgeted_agent.trace import record_episode, replay


def show(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    show("1. complementary_bits: optimal budget planning beats one-step greedy")
    for kappa in (F(0), F(1, 16)):
        p = examples.complementary_bits(kappa)
        pl = OptimalPlanner(p)
        v_opt = pl.value()
        print(f"kappa={kappa}")
        print(f"  immediate stop        : {immediate_stop_value(p)}")
        print(f"  fixed [pair_id]       : {fixed_order_value(p, [0])}")
        print(f"  fixed [parity,high]   : {fixed_order_value(p, [1, 2])}")
        print(f"  one-step greedy       : {policy_value(p, greedy_policy(p))}")
        print(f"  OPTIMAL               : {v_opt}  (root action: tool {pl.policy()})")
        print(f"  planner stats         : {pl.stats()}")
    p = examples.complementary_bits()
    best, _, n_pol = enumerate_all_policies(p)
    print(f"  brute-force check     : max over all {n_pol} policies = {best}")
    pl = OptimalPlanner(p)
    by_paths, n_paths = policy_value_by_paths(p, pl.policy)
    print(f"  path-sum of optimal policy over {n_paths} paths = {by_paths}")

    show("2. eig_trap: max information gain != max final decision value")
    p = examples.eig_trap()
    pl = OptimalPlanner(p)
    print(f"  EIG(tail_id)     = {expected_information_gain(p, p.prior, 0):.6f} bits")
    print(f"  EIG(leader_test) = {expected_information_gain(p, p.prior, 1):.6f} bits")
    print(f"  value(tail_id)   = {policy_value(p, lambda b, B: 0 if B >= 1 else None)}")
    print(f"  value(leader)    = {policy_value(p, lambda b, B: 1 if B >= 1 else None)}")
    print(f"  OPTIMAL          = {pl.value()} (picks tool {pl.policy()}, the lower-EIG one)")

    show("3. rounding_trap: approximate cache merging flips the policy")
    p = examples.rounding_trap_problem()
    exact = OptimalPlanner(p)
    print(f"  exact  : below -> {exact.policy(examples.ROUNDING_BELOW, 1)}, "
          f"above -> {exact.policy(examples.ROUNDING_ABOVE, 1)}")
    rounded = RoundedCachePlanner(p, ndigits=2)
    rounded.value(examples.ROUNDING_BELOW, 1)
    print(f"  rounded: below -> {rounded.policy(examples.ROUNDING_BELOW, 1)}, "
          f"above -> {rounded.policy(examples.ROUNDING_ABOVE, 1)}  (WRONG for above)")

    show("4. noisy_channel: fixed-seed Monte Carlo vs exact")
    p = examples.noisy_channel()
    pl = OptimalPlanner(p)
    exact_v = pl.value()
    print(f"  exact optimal = {exact_v} = {float(exact_v)}")
    for name, factory in (
        ("optimal", lambda: pl.policy),
        ("greedy", lambda: greedy_policy(p)),
        ("fixed[0,0]", lambda: FixedOrderPolicy(p, [0, 0])),
    ):
        sim = simulate(p, factory, episodes=4000, seed=20260930)
        print(f"  MC {name:12s}: mean={sim.mean:.4f} +- {sim.std_err:.4f} "
              f"(seed={sim.seed}, n={sim.episodes})")

    show("5. state-space growth: cached states vs tree paths")
    for budget in (2, 3, 4, 5, 6):
        p = examples.noisy_channel(budget=budget)
        pl = OptimalPlanner(p)
        pl.value()
        print(f"  budget={budget}: cached states={pl.stats()['cache_size']:4d} "
              f"reachable={len(reachable_states(p)):4d} "
              f"tree paths={count_tree_paths(p)}")

    show("6. replayable episode record (fully recomputable from inputs)")
    p = examples.noisy_channel()
    pl = OptimalPlanner(p)
    pl.value()
    rec = record_episode(p, pl.policy, z=0, rng=random.Random(123))
    out = replay(rec)
    print(json.dumps(rec, indent=2))
    print(f"  replayed: spent={out['spent']} final_belief={out['final_belief']} "
          f"guess={out['guess']} payoff={out['payoff']}")

    show("7. correlated calls: (belief, budget) is no longer sufficient")
    demo = examples.correlation_demo()
    for k, v in demo.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
