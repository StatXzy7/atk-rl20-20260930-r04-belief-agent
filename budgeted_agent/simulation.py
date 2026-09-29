"""Fixed-seed Monte Carlo simulation of policies.

The simulator samples a true z from the prior and observations from
P(o | z, a); the policy only ever sees the belief/budget state, never z.
Floats are used here (simulation is statistical, not the exact reference).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .belief import best_guess, update
from .model import Problem
from .planner import STOP


def sample_categorical(weights, rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, w in enumerate(weights):
        acc += float(w)
        if r < acc:
            return i
    return len(weights) - 1


def run_episode(problem: Problem, policy_fn, z: int, rng: random.Random) -> float:
    belief = problem.prior
    budget = problem.budget
    spent = 0
    while True:
        a = policy_fn(belief, budget)
        if a is STOP:
            break
        t = problem.tools[a]
        o = sample_categorical(t.likelihoods[z], rng)
        belief = update(belief, t, o)
        budget -= t.cost
        spent += t.cost
    correct = 1 if best_guess(belief) == z else 0
    return correct - float(problem.kappa) * spent


@dataclass(frozen=True)
class SimResult:
    episodes: int
    seed: int
    mean: float
    std_err: float


def simulate(problem: Problem, policy_factory, episodes: int, seed: int) -> SimResult:
    """policy_factory() must return a fresh callable per episode (so stateful
    policies such as FixedOrderPolicy can reset their pointer)."""
    rng = random.Random(seed)
    total = 0.0
    total_sq = 0.0
    for _ in range(episodes):
        z = sample_categorical(problem.prior, rng)
        x = run_episode(problem, policy_factory(), z, rng)
        total += x
        total_sq += x * x
    mean = total / episodes
    var = total_sq / episodes - mean * mean
    var = max(var, 0.0)
    se = math.sqrt(var / episodes)
    return SimResult(episodes=episodes, seed=seed, mean=mean, std_err=se)
