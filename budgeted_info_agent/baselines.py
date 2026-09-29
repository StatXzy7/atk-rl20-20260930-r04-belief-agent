"""Baselines: immediate stop, fixed tool order, one-step greedy value.

Also provides `policy_value`, the exact expected payoff of any memoryless
policy, computed by exact enumeration over observation paths.
"""
from __future__ import annotations

from fractions import Fraction

from .belief import Belief, best_guess, outcome_prob, stop_value, update
from .model import Problem
from .planner import CALL, STOP, Action


def immediate_stop(problem: Problem) -> tuple[Fraction, Action]:
    """Baseline 1: stop at once, report the prior argmax."""
    return stop_value(problem.prior), (STOP, best_guess(problem.prior))


def greedy_action(problem: Problem, belief: Belief, budget: int) -> Action:
    """Baseline 3 (myopic): compare stopping against calling one tool and
    then stopping. Ties resolve to stopping (strict improvement required)."""
    best = stop_value(belief)
    action: Action = (STOP, best_guess(belief))
    for tool_index, tool in enumerate(problem.tools):
        if tool.cost > budget:
            continue
        q = -problem.kappa * tool.cost
        for o in range(tool.n_outcomes):
            p_o = outcome_prob(belief, tool, o)
            if p_o == 0:
                continue
            q += p_o * stop_value(update(belief, tool, o))
        if q > best:
            best = q
            action = (CALL, tool_index)
    return action


def greedy_policy(problem: Problem):
    """One-step lookahead policy: re-evaluates `greedy_action` at every
    reached state. Its *true* expected payoff must be computed by rolling
    it out (see `policy_value`), not by its own myopic estimate."""

    def policy(belief: Belief, budget: int) -> Action:
        return greedy_action(problem, belief, budget)

    return policy


class FixedOrderPolicy:
    """Baseline 2: call tools in a fixed order while affordable, then stop.

    Stateful (remembers the position in the order), so build a fresh
    instance per episode via `reset()`.
    """

    def __init__(self, problem: Problem, order):
        self.problem = problem
        self.order = tuple(order)
        self._pos = 0

    def reset(self) -> "FixedOrderPolicy":
        self._pos = 0
        return self

    def __call__(self, belief: Belief, budget: int) -> Action:
        if self._pos < len(self.order):
            tool_index = self.order[self._pos]
            tool = self.problem.tools[tool_index]
            if tool.cost <= budget:
                self._pos += 1
                return (CALL, tool_index)
        return (STOP, best_guess(belief))


def fixed_order_value(
    problem: Problem,
    order,
    belief: Belief | None = None,
    budget: int | None = None,
    pos: int = 0,
) -> Fraction:
    """Exact expected payoff of the fixed-order policy (path enumeration)."""
    if belief is None:
        belief, budget = problem.prior, problem.budget
    if pos < len(order):
        tool = problem.tools[order[pos]]
        if tool.cost <= budget:
            total = -problem.kappa * tool.cost
            for o in range(tool.n_outcomes):
                p_o = outcome_prob(belief, tool, o)
                if p_o == 0:
                    continue
                total += p_o * fixed_order_value(
                    problem, order, update(belief, tool, o),
                    budget - tool.cost, pos + 1,
                )
            return total
    return stop_value(belief)


def policy_value(
    problem: Problem,
    policy,
    belief: Belief | None = None,
    budget: int | None = None,
) -> Fraction:
    """Exact expected payoff of a memoryless policy, by enumerating all
    observation paths with their exact probabilities."""
    if belief is None:
        belief, budget = problem.prior, problem.budget
    action = policy(belief, budget)
    if action[0] == STOP:
        return stop_value(belief)
    tool = problem.tools[action[1]]
    if tool.cost > budget:
        raise ValueError("policy called a tool the budget cannot afford")
    total = -problem.kappa * tool.cost
    for o in range(tool.n_outcomes):
        p_o = outcome_prob(belief, tool, o)
        if p_o == 0:
            continue
        total += p_o * policy_value(
            problem, policy, update(belief, tool, o), budget - tool.cost
        )
    return total
