"""Seeded simulation. The policy callable receives exactly (belief, budget);
the hidden state z lives only inside the episode runner and is used solely
to draw observations and to score the final report. Every episode record
can be fully recomputed from the problem inputs (see `replay_episode`).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction

from .belief import Belief, update
from .model import Problem
from .planner import STOP


@dataclass
class StepRecord:
    tool_index: int
    outcome_index: int
    belief_after: Belief
    cost: int


@dataclass
class EpisodeRecord:
    z: int
    steps: list = field(default_factory=list)
    guess: int = -1
    correct: bool = False
    tool_cost_total: int = 0
    reward: Fraction = Fraction(0)


def _sample(rng: random.Random, probs) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += float(p)
        if r < acc:
            return i
    return len(probs) - 1


def run_episode(problem: Problem, policy, rng: random.Random) -> EpisodeRecord:
    z = _sample(rng, problem.prior)  # hidden from the policy
    belief, budget = problem.prior, problem.budget
    record = EpisodeRecord(z=z)
    while True:
        action = policy(belief, budget)  # policy never sees z
        if action[0] == STOP:
            record.guess = action[1]
            record.correct = record.guess == z
            record.reward = Fraction(1 if record.correct else 0) - (
                problem.kappa * record.tool_cost_total
            )
            return record
        tool = problem.tools[action[1]]
        if tool.cost > budget:
            raise ValueError("policy called a tool the budget cannot afford")
        outcome = _sample(rng, tool.likelihood[z])
        belief = update(belief, tool, outcome)
        budget -= tool.cost
        record.tool_cost_total += tool.cost
        record.steps.append(StepRecord(action[1], outcome, belief, tool.cost))


def simulate(problem: Problem, policy_factory, n_episodes: int, seed: int):
    """Run `n_episodes` episodes under a fresh policy instance each.

    Returns exact rational mean reward plus a float standard error, and the
    full episode records. Fixed seed => fully reproducible output."""
    rng = random.Random(seed)
    records = [
        run_episode(problem, policy_factory(), rng) for _ in range(n_episodes)
    ]
    mean = sum((r.reward for r in records), Fraction(0)) / n_episodes
    xs = [float(r.reward) for r in records]
    m = sum(xs) / len(xs)
    if len(xs) > 1:
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    else:
        var = 0.0
    stderr = (var / len(xs)) ** 0.5
    return {"mean": mean, "stderr": stderr, "records": records}


def replay_episode(problem: Problem, record: EpisodeRecord) -> Fraction:
    """Recompute beliefs, costs and reward purely from the problem inputs
    and the recorded (tool, outcome) sequence. Raises on any mismatch, so
    a record is a self-auditing trace."""
    belief = problem.prior
    total_cost = 0
    for step in record.steps:
        tool = problem.tools[step.tool_index]
        belief = update(belief, tool, step.outcome_index)
        total_cost += tool.cost
        assert belief == step.belief_after, "posterior mismatch"
        assert step.cost == tool.cost, "cost mismatch"
    assert total_cost == record.tool_cost_total, "total cost mismatch"
    reward = Fraction(1 if record.guess == record.z else 0) - (
        problem.kappa * total_cost
    )
    assert reward == record.reward, "reward mismatch"
    return reward
