"""Expected information gain (entropy reduction), for comparison only.

The planner optimizes expected FINAL DECISION VALUE, not information gain.
This module exists to demonstrate (in examples.eig_trap) that the two
criteria can rank tools differently. Entropy uses floats; decision values
stay exact.
"""
from __future__ import annotations

import math

from .belief import marginal, update
from .model import Problem


def entropy(belief) -> float:
    h = 0.0
    for b in belief:
        if b > 0:
            p = float(b)
            h -= p * math.log2(p)
    return h


def expected_information_gain(problem: Problem, belief, tool_index: int) -> float:
    t = problem.tools[tool_index]
    post_h = 0.0
    for o in range(t.n_outcomes):
        p = marginal(belief, t, o)
        if p:
            post_h += float(p) * entropy(update(belief, t, o))
    return entropy(belief) - post_h
