"""Information-gain heuristic (float, heuristic use only).

Expected entropy reduction is a legitimate scoring rule for some purposes,
but it is NOT the final-decision value: the Bellman objective pays for
max_z of the posterior, not for entropy. See the trap example in
examples.infogain_trap_example.
"""
from __future__ import annotations

import math

from .belief import Belief, outcome_prob, stop_value, update
from .model import Problem


def entropy(belief: Belief) -> float:
    return -sum(
        float(b) * math.log2(float(b)) for b in belief if b > 0
    )


def expected_info_gain(problem: Problem, belief: Belief, tool_index: int) -> float:
    tool = problem.tools[tool_index]
    gain = entropy(belief)
    for o in range(tool.n_outcomes):
        p_o = outcome_prob(belief, tool, o)
        if p_o == 0:
            continue
        gain -= float(p_o) * entropy(update(belief, tool, o))
    return gain


def expected_decision_value(problem: Problem, belief: Belief, tool_index: int):
    """E[max_z b'(z)] after one call, minus kappa * cost: the one-step
    quantity that actually enters the Bellman objective."""
    tool = problem.tools[tool_index]
    total = -problem.kappa * tool.cost
    for o in range(tool.n_outcomes):
        p_o = outcome_prob(belief, tool, o)
        if p_o == 0:
            continue
        total += p_o * stop_value(update(belief, tool, o))
    return total
