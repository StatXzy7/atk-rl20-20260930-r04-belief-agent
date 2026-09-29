"""Problem definition: hidden variable, tools, costs, budget, cost rate kappa.

Everything is exact: probabilities are fractions.Fraction. Zero-probability
likelihood entries and degenerate (one-hot) priors are explicitly allowed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction

Belief = tuple  # tuple[Fraction, ...], one entry per hidden state z


def _frac(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        return Fraction(x)
    raise TypeError(f"cannot parse an exact fraction from {x!r}")


@dataclass(frozen=True)
class Tool:
    """A tool with positive integer cost and likelihoods[z][o] = P(o | z, tool)."""

    name: str
    cost: int
    likelihoods: tuple  # tuple[tuple[Fraction, ...], ...]

    @property
    def n_outcomes(self) -> int:
        return len(self.likelihoods[0])


@dataclass(frozen=True)
class Problem:
    prior: Belief
    tools: tuple  # tuple[Tool, ...]
    kappa: Fraction
    budget: int

    def __post_init__(self):
        object.__setattr__(self, "prior", tuple(_frac(x) for x in self.prior))
        object.__setattr__(self, "kappa", _frac(self.kappa))
        object.__setattr__(self, "budget", int(self.budget))
        object.__setattr__(
            self,
            "tools",
            tuple(
                Tool(
                    t.name,
                    int(t.cost),
                    tuple(tuple(_frac(v) for v in row) for row in t.likelihoods),
                )
                for t in self.tools
            ),
        )
        self._validate()

    @property
    def n_states(self) -> int:
        return len(self.prior)

    def _validate(self):
        n = len(self.prior)
        if n == 0:
            raise ValueError("prior must be non-empty")
        if any(p < 0 for p in self.prior):
            raise ValueError("prior entries must be >= 0")
        if sum(self.prior) != 1:
            raise ValueError("prior must sum to 1")
        if self.budget < 0:
            raise ValueError("budget must be >= 0")
        if self.kappa < 0:
            raise ValueError("kappa must be >= 0")
        for t in self.tools:
            if t.cost < 1:
                raise ValueError("tool cost must be a positive integer")
            if len(t.likelihoods) != n:
                raise ValueError("each tool needs one likelihood row per hidden state")
            if t.n_outcomes == 0:
                raise ValueError("each tool needs at least one outcome")
            for row in t.likelihoods:
                if len(row) != t.n_outcomes:
                    raise ValueError("ragged likelihood table")
                if any(v < 0 for v in row):
                    raise ValueError("negative likelihood")
                if sum(row) != 1:
                    raise ValueError("each likelihood row must sum to 1")


def problem_to_dict(problem: Problem) -> dict:
    return {
        "prior": [str(x) for x in problem.prior],
        "kappa": str(problem.kappa),
        "budget": problem.budget,
        "tools": [
            {
                "name": t.name,
                "cost": t.cost,
                "likelihoods": [[str(v) for v in row] for row in t.likelihoods],
            }
            for t in problem.tools
        ],
    }


def problem_from_dict(d: dict) -> Problem:
    return Problem(
        prior=tuple(d["prior"]),
        kappa=d["kappa"],
        budget=d["budget"],
        tools=tuple(
            Tool(t["name"], t["cost"], tuple(tuple(row) for row in t["likelihoods"]))
            for t in d["tools"]
        ),
    )


def dumps(problem: Problem) -> str:
    return json.dumps(problem_to_dict(problem), indent=2, sort_keys=True)


def loads(text: str) -> Problem:
    return problem_from_dict(json.loads(text))
