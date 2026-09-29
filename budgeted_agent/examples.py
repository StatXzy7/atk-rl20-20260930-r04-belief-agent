"""Named examples used by tests, the demo, and the README derivations.

Every constant is an exact Fraction so all reported values are recomputable
by hand from this file.
"""
from __future__ import annotations

from fractions import Fraction as F

from .model import Problem, Tool


def complementary_bits(kappa=F(0)) -> Problem:
    """4 equally likely states. pair_id (cost 2) identifies z in {0,1} and
    merges {2,3}; parity (cost 1) and high_bit (cost 1) each reveal one bit.
    Budget 2.

    One-step greedy takes pair_id (one-step value 3/4 > 1/2 for each bit)
    and finishes at 3/4. The optimal plan buys both bits (1 + 1 budget)
    and reaches certainty: value 1 (with kappa=0).
    """
    prior = (F(1, 4),) * 4
    pair_id = Tool(
        "pair_id",
        2,
        (
            (F(1), F(0), F(0)),
            (F(0), F(1), F(0)),
            (F(0), F(0), F(1)),
            (F(0), F(0), F(1)),
        ),
    )
    parity = Tool(
        "parity",
        1,
        ((F(1), F(0)), (F(0), F(1)), (F(1), F(0)), (F(0), F(1))),
    )
    high_bit = Tool(
        "high_bit",
        1,
        ((F(1), F(0)), (F(1), F(0)), (F(0), F(1)), (F(0), F(1))),
    )
    return Problem(prior, (pair_id, parity, high_bit), kappa, 2)


def eig_trap() -> Problem:
    """Max-entropy-gain tool choice is NOT max-decision-value tool choice.

    Prior (1/2, 1/8, 1/8, 1/8, 1/8), budget 1, kappa 0.
    - tail_id (cost 1): if z=0 outputs a uniform fake in {1..4}, else names z.
      EIG = 1 bit, but the posterior is always (1/2, 1/2, 0,0,0)-shaped, so
      expected max posterior = 1/2 (no decision gain).
    - leader_test (cost 1): detects z=0 with P(yes|z=0)=9/10, never fires else.
      EIG ~ 0.759 bits < 1 bit, but expected max posterior = 23/40 > 1/2.
    Max-EIG picks tail_id and loses 3/40 of expected decision value.
    """
    prior = (F(1, 2),) + (F(1, 8),) * 4
    tail_id = Tool(
        "tail_id",
        1,
        (
            (F(1, 4), F(1, 4), F(1, 4), F(1, 4)),
            (F(1), F(0), F(0), F(0)),
            (F(0), F(1), F(0), F(0)),
            (F(0), F(0), F(1), F(0)),
            (F(0), F(0), F(0), F(1)),
        ),
    )
    leader_test = Tool(
        "leader_test",
        1,
        (
            (F(9, 10), F(1, 10)),
            (F(0), F(1)),
            (F(0), F(1)),
            (F(0), F(1)),
            (F(0), F(1)),
        ),
    )
    return Problem(prior, (tail_id, leader_test), F(0), 1)


def rounding_trap_problem() -> Problem:
    """Symmetric binary channel (accuracy 3/4), cost 1, kappa 1/10, budget 1.

    For a belief b0 = P(z=0) in [1/2, 3/4], E[max posterior after one call]
    is constantly 3/4, so calling is optimal iff 3/4 - 1/10 > b0, i.e.
    b0 < 13/20. The two beliefs ROUNDING_BELOW=649/1000 and
    ROUNDING_ABOVE=651/1000 sit on opposite sides of the threshold but share
    the 2-decimal rounded key (0.65, 0.35): a rounding cache merges them and
    must get one of them wrong.
    """
    chan = Tool("chan", 1, ((F(3, 4), F(1, 4)), (F(1, 4), F(3, 4))))
    return Problem((F(1, 2), F(1, 2)), (chan,), F(1, 10), 1)


ROUNDING_BELOW = (F(649, 1000), F(351, 1000))
ROUNDING_ABOVE = (F(651, 1000), F(349, 1000))


def noisy_channel(kappa=F(1, 10), budget=2) -> Problem:
    """2 states, uniform prior, one symmetric 3/4-accuracy channel (cost 1).

    Exact optimum with budget 2 and kappa=1/10: call once, then stop.
    V = 3/4 - 1/10 = 13/20. Used for Monte Carlo vs exact comparison.
    """
    chan = Tool("chan", 1, ((F(3, 4), F(1, 4)), (F(1, 4), F(3, 4))))
    return Problem((F(1, 2), F(1, 2)), (chan,), kappa, budget)


def adaptive_savings() -> Problem:
    """Adaptivity beats any fixed order once calls cost money.

    Prior (1/2, 1/4, 1/4), kappa 1/16, budget 2.
    - first (cost 1): o=0 iff z=0 (sometimes certifies z in one call).
    - tail (cost 1): z=0 mimics {1,2} uniformly; z=1 -> o0; z=2 -> o1.
    Optimal: call first; stop if o=0, else call tail. V = 29/32.
    Fixed order [first, tail] always spends 2: V = 7/8 = 28/32.
    """
    prior = (F(1, 2), F(1, 4), F(1, 4))
    first = Tool("first", 1, ((F(1), F(0)), (F(0), F(1)), (F(0), F(1))))
    tail = Tool(
        "tail",
        1,
        ((F(1, 2), F(1, 2)), (F(1), F(0)), (F(0), F(1))),
    )
    return Problem(prior, (first, tail), F(1, 16), 2)


def correlation_demo():
    """Why (belief, budget) is not a sufficient state once tool calls are
    correlated given z.

    z in {0,1}, prior (2/3, 1/3). One tool, called twice; joint distribution
    of the two outputs given z: o1 is pure noise (fair coin), and
    o2 = z if o1 == 0 else 1 - z. Both one-step histories h0=(o1=0) and
    h1=(o1=1) yield the SAME belief (2/3, 1/3), yet the predictive
    distribution of the next output differs:
    P(o2=0 | h0) = 2/3,  P(o2=0 | h1) = 1/3.
    Returns exact quantities computed from the joint table.
    """
    prior = (F(2, 3), F(1, 3))
    joint = (
        ((F(1, 2), F(0)), (F(0), F(1, 2))),
        ((F(0), F(1, 2)), (F(1, 2), F(0))),
    )

    def belief_after(o1):
        num = tuple(prior[z] * sum(joint[z][o1]) for z in (0, 1))
        m = sum(num)
        return tuple(x / m for x in num)

    def predictive_o2_0(o1):
        num = sum(prior[z] * joint[z][o1][0] for z in (0, 1))
        den = sum(prior[z] * sum(joint[z][o1]) for z in (0, 1))
        return num / den

    return {
        "prior": prior,
        "belief_after_o1_eq_0": belief_after(0),
        "belief_after_o1_eq_1": belief_after(1),
        "P_o2_eq_0_given_o1_eq_0": predictive_o2_0(0),
        "P_o2_eq_0_given_o1_eq_1": predictive_o2_0(1),
    }
