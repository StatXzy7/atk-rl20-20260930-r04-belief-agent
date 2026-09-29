"""Belief-state mechanics: Bayesian update, outcome prediction, stopping value.

A belief is a tuple of Fractions summing to 1. All arithmetic is exact.
"""
from __future__ import annotations

from fractions import Fraction

from .model import Tool

Belief = tuple[Fraction, ...]


def outcome_prob(belief: Belief, tool: Tool, outcome: int) -> Fraction:
    """P(o | b, a) = sum_z b(z) * P(o | z, a)."""
    return sum(b * tool.likelihood[z][outcome] for z, b in enumerate(belief))


def update(belief: Belief, tool: Tool, outcome: int) -> Belief | None:
    """Posterior b'(z) = b(z) P(o|z,a) / P(o|b,a).

    Returns None when the outcome has zero probability under the belief
    (an impossible observation); callers must prune such branches instead
    of dividing by zero.
    """
    p = outcome_prob(belief, tool, outcome)
    if p == 0:
        return None
    return tuple(
        b * tool.likelihood[z][outcome] / p for z, b in enumerate(belief)
    )


def stop_value(belief: Belief) -> Fraction:
    """Value of stopping now: probability that the best report is correct."""
    return max(belief)


def best_guess(belief: Belief) -> int:
    """Bayes-optimal report: argmax_z b(z), ties broken by smallest index
    so decisions are deterministic and reproducible."""
    return max(range(len(belief)), key=lambda z: (belief[z], -z))
