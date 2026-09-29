"""Exact Bayesian belief operations.

Bayesian update (conditional independence of calls given z and the tool):

    b'(z) = b(z) * P(o | z, a) / P(o | b, a),
    P(o | b, a) = sum_z b(z) * P(o | z, a).

The update is only defined when P(o | b, a) > 0; callers must skip
zero-probability observations (they occur with probability zero).
"""
from __future__ import annotations

from fractions import Fraction

from .model import Tool


def marginal(belief, tool: Tool, outcome: int) -> Fraction:
    """P(o | b, a) = sum_z b(z) P(o | z, a)."""
    return sum(b * row[outcome] for b, row in zip(belief, tool.likelihoods))


def update(belief, tool: Tool, outcome: int):
    """Posterior after observing `outcome` from `tool` under `belief`."""
    m = marginal(belief, tool, outcome)
    if m == 0:
        raise ValueError("cannot condition on a zero-probability observation")
    return tuple(b * row[outcome] / m for b, row in zip(belief, tool.likelihoods))


def stop_value(belief) -> Fraction:
    """Value of stopping: report argmax_z b(z), expected reward = max_z b(z)."""
    return max(belief)


def best_guess(belief) -> int:
    """Deterministic argmax (lowest index wins ties)."""
    best_i, best_v = 0, belief[0]
    for i, v in enumerate(belief):
        if v > best_v:
            best_i, best_v = i, v
    return best_i
