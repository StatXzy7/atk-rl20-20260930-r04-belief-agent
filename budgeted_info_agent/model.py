"""Problem definition for the finite-budget information acquisition agent.

All probabilities and the cost coefficient kappa are exact rationals
(`fractions.Fraction`). The hidden state z is never part of any policy input;
it appears only in the simulator for scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Tool:
    """A tool with positive integer cost and a finite output alphabet.

    likelihood[z][o] = P(o | z, this tool). Outputs of different calls are
    conditionally independent given z and the chosen tools.
    """

    name: str
    cost: int
    outcome_names: tuple[str, ...]
    likelihood: tuple[tuple[Fraction, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cost, int) or self.cost <= 0:
            raise ValueError("tool cost must be a positive integer")
        if not self.outcome_names:
            raise ValueError("tool must have at least one outcome")
        for row in self.likelihood:
            if len(row) != len(self.outcome_names):
                raise ValueError("likelihood row size must match outcome count")
            if any(p < 0 or p > 1 for p in row):
                raise ValueError("likelihoods must be probabilities in [0, 1]")
            if sum(row) != 1:
                raise ValueError("each conditional distribution must sum to 1")

    @property
    def n_outcomes(self) -> int:
        return len(self.outcome_names)


@dataclass(frozen=True)
class Problem:
    """Full problem instance: prior, tools, cost coefficient, initial budget."""

    state_names: tuple[str, ...]
    prior: tuple[Fraction, ...]
    tools: tuple[Tool, ...]
    kappa: Fraction
    budget: int

    def __post_init__(self) -> None:
        if not self.state_names:
            raise ValueError("need at least one hidden state")
        if len(self.prior) != len(self.state_names):
            raise ValueError("prior size must match state count")
        if any(p < 0 for p in self.prior) or sum(self.prior) != 1:
            raise ValueError("prior must be a probability distribution")
        if self.kappa < 0:
            raise ValueError("kappa must be >= 0")
        if not isinstance(self.budget, int) or self.budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        for tool in self.tools:
            if len(tool.likelihood) != len(self.state_names):
                raise ValueError("likelihood must have one row per hidden state")

    @property
    def n_states(self) -> int:
        return len(self.state_names)
