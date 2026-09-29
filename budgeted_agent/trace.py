"""Episode records and exact replay.

A record contains only externally logged data: the problem spec, the
sequence of (tool, observation) pairs, the reported guess, and (for payoff
scoring) the true z. replay() recomputes every posterior, the spent cost,
and the payoff purely from the record, so any reported episode is fully
auditable.
"""
from __future__ import annotations

import random
from fractions import Fraction

from .belief import best_guess, marginal, update
from .model import Problem, problem_from_dict, problem_to_dict
from .planner import STOP
from .simulation import sample_categorical


def record_episode(problem: Problem, policy_fn, z: int, rng: random.Random) -> dict:
    """Run a memoryless policy against true state z and log what happened."""
    belief = problem.prior
    budget = problem.budget
    steps = []
    while True:
        a = policy_fn(belief, budget)
        if a is STOP:
            break
        t = problem.tools[a]
        o = sample_categorical(t.likelihoods[z], rng)
        steps.append({"tool": a, "observation": o})
        belief = update(belief, t, o)
        budget -= t.cost
    return {
        "problem": problem_to_dict(problem),
        "steps": steps,
        "guess": best_guess(belief),
        "true_z": z,
    }


def replay(record: dict) -> dict:
    """Recompute beliefs, costs and payoff from a record. Raises on any
    inconsistency (budget violation or zero-probability observation)."""
    problem = problem_from_dict(record["problem"])
    belief = problem.prior
    spent = 0
    log = []
    for step in record["steps"]:
        a, o = step["tool"], step["observation"]
        t = problem.tools[a]
        if t.cost > problem.budget - spent:
            raise ValueError("budget violated by record")
        if marginal(belief, t, o) == 0:
            raise ValueError("zero-probability observation in record")
        belief = update(belief, t, o)
        spent += t.cost
        log.append(
            {
                "tool": t.name,
                "observation": o,
                "posterior": tuple(str(x) for x in belief),
                "spent": spent,
            }
        )
    guess = record["guess"]
    correct = 1 if guess == record["true_z"] else 0
    payoff = Fraction(correct) - problem.kappa * spent
    return {
        "log": log,
        "spent": spent,
        "final_belief": belief,
        "guess": guess,
        "correct": bool(correct),
        "payoff": payoff,
    }
