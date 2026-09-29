"""Worked examples. Every number here is an exact rational, so all reported
payoffs can be recomputed by hand or by the independent enumerators.
"""
from __future__ import annotations

from fractions import Fraction as F

from .model import Problem, Tool


def parity_example() -> Problem:
    """Two deterministic tools, each revealing one bit of a 4-state variable.

    prior uniform over {00, 01, 10, 11}; kappa = 1/10; each call costs 3;
    budget 6. One call alone never pays (1/2 - 3/10 = 1/5 < 1/4), so
    one-step greedy stops immediately with value 1/4. Two calls identify z
    exactly: 1 - 6/10 = 2/5. Optimal (2/5) strictly beats greedy (1/4).
    """
    zero, one = F(0), F(1)
    bit1 = Tool(
        name="reveal-bit1",
        cost=3,
        outcome_names=("b1=0", "b1=1"),
        likelihood=((one, zero), (one, zero), (zero, one), (zero, one)),
    )
    bit2 = Tool(
        name="reveal-bit2",
        cost=3,
        outcome_names=("b2=0", "b2=1"),
        likelihood=((one, zero), (zero, one), (one, zero), (zero, one)),
    )
    return Problem(
        state_names=("z00", "z01", "z10", "z11"),
        prior=(F(1, 4),) * 4,
        tools=(bit1, bit2),
        kappa=F(1, 10),
        budget=6,
    )


def bsc_example(budget: int = 2) -> Problem:
    """Binary z, one symmetric channel with accuracy 3/4, cost 1,
    kappa = 1/10. Exact optimum for budget >= 1 is 13/20: call once, then
    stop (a second call adds no decision value here)."""
    p, q = F(3, 4), F(1, 4)
    tool = Tool(
        name="bsc-3/4",
        cost=1,
        outcome_names=("o=0", "o=1"),
        likelihood=((p, q), (q, p)),
    )
    return Problem(
        state_names=("z0", "z1"),
        prior=(F(1, 2), F(1, 2)),
        tools=(tool,),
        kappa=F(1, 10),
        budget=budget,
    )


def infogain_trap_example() -> Problem:
    """Tool A maximizes expected entropy reduction; tool B maximizes expected
    decision value. They rank oppositely, proving that information gain is
    not final-decision value.

    prior = (9/20, 7/20, 1/40 x8). Tool A resolves the 8 tail states
    exactly but merges the top two; tool B separates z0 from the rest.
    Expected info gain: A > B. Expected max posterior: A = 13/20 < B = 4/5.
    """
    zero, one = F(0), F(1)
    n_tail = 8
    states = ("z0", "z1") + tuple(f"t{i}" for i in range(n_tail))
    prior = (F(9, 20), F(7, 20)) + (F(1, 40),) * n_tail

    tail_rows = []
    for z in range(10):
        if z < 2:
            tail_rows.append((one,) + (zero,) * n_tail)
        else:
            tail_rows.append((zero,) * (z - 1) + (one,) + (zero,) * (9 - z))
    tool_a = Tool(
        name="resolve-tail",
        cost=1,
        outcome_names=("top",) + tuple(f"t{i}" for i in range(n_tail)),
        likelihood=tuple(tail_rows),
    )
    tool_b = Tool(
        name="separate-z0",
        cost=1,
        outcome_names=("is-z0", "not-z0"),
        likelihood=((one, zero),) + ((zero, one),) * 9,
    )
    return Problem(
        state_names=states,
        prior=prior,
        tools=(tool_a, tool_b),
        kappa=F(0),
        budget=1,
    )


def correlation_example() -> dict:
    """Same per-tool likelihoods, two different joint worlds.

    Three calls of a 3/4-accuracy binary tool, kappa = 0, budget 3.
    World IND: outputs conditionally independent given z (planner's model).
    World COR: noise is perfectly shared, so all three outputs coincide.
    The planner's state (belief, budget) cannot distinguish the worlds:
    it predicts 27/32 for both, while the true correlated value is 3/4.
    """
    p, q = F(3, 4), F(1, 4)
    tool = Tool(
        name="bsc-3/4",
        cost=1,
        outcome_names=("o=0", "o=1"),
        likelihood=((p, q), (q, p)),
    )
    problem = Problem(
        state_names=("z0", "z1"),
        prior=(F(1, 2), F(1, 2)),
        tools=(tool,),
        kappa=F(0),
        budget=3,
    )
    # joint[(z, (o1, o2, o3))] = P(o1, o2, o3 | z) under perfect correlation
    joint = {}
    for z in (0, 1):
        for outs in ((0, 0, 0), (1, 1, 1)):
            correct = outs[0] == z
            joint[(z, outs)] = p if correct else q
    return {"problem": problem, "correlated_joint": joint}


def correlated_world_value(prior, joint, n_calls: int) -> F:
    """Exact optimal value in the correlated world: with kappa = 0 the
    optimum uses all calls, so value = sum_outcomes max_z prior[z] P(o|z)."""
    import itertools

    value = F(0)
    for outs in itertools.product((0, 1), repeat=n_calls):
        value += max(
            prior[z] * joint.get((z, outs), F(0)) for z in range(len(prior))
        )
    return value
