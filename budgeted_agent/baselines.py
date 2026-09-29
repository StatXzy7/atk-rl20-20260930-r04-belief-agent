"""Baseline policies and exact expected-value evaluation.

All values are exact Fractions. A "memoryless" policy maps (belief, budget)
to an action; FixedOrderPolicy is history-dependent (it carries a pointer
into its tool list), so its exact value is computed by a dedicated recursion
over (position, belief, budget).
"""
from __future__ import annotations

from fractions import Fraction

from .belief import marginal, stop_value, update
from .model import Problem
from .planner import STOP


def immediate_stop_value(problem: Problem, belief=None) -> Fraction:
    """Baseline 1: stop immediately and report the prior MAP hypothesis."""
    return stop_value(problem.prior if belief is None else belief)


class FixedOrderPolicy:
    """Baseline 2: try tools in a fixed order; call the next one whose cost
    fits the remaining budget (unaffordable ones are skipped); stop when the
    list is exhausted."""

    def __init__(self, problem: Problem, order):
        self.problem = problem
        self.order = tuple(order)
        self._ptr = 0

    def reset(self):
        self._ptr = 0

    def __call__(self, belief, budget):
        while self._ptr < len(self.order):
            a = self.order[self._ptr]
            self._ptr += 1
            if self.problem.tools[a].cost <= budget:
                return a
        return STOP


def fixed_order_value(problem: Problem, order, belief=None, budget=None) -> Fraction:
    """Exact expected value of the fixed-order policy."""
    belief = problem.prior if belief is None else belief
    budget = problem.budget if budget is None else budget
    order = tuple(order)
    memo = {}

    def rec(i, b, B):
        key = (i, b, B)
        if key in memo:
            return memo[key]
        if i == len(order):
            v = stop_value(b)
        else:
            t = problem.tools[order[i]]
            if t.cost > B:
                v = rec(i + 1, b, B)
            else:
                v = -problem.kappa * t.cost
                for o in range(t.n_outcomes):
                    p = marginal(b, t, o)
                    if p:
                        v += p * rec(i + 1, update(b, t, o), B - t.cost)
        memo[key] = v
        return v

    return rec(0, belief, budget)


def greedy_action(problem: Problem, belief, budget):
    """Baseline 3 (one-step greedy value): pick the action maximizing
    -kappa * cost + E[ max_z b'(z) ], i.e. the value of calling the tool once
    and then being forced to stop. STOP wins ties."""
    best = stop_value(belief)
    act = STOP
    for ai, t in enumerate(problem.tools):
        if t.cost > budget:
            continue
        q = -problem.kappa * t.cost
        for o in range(t.n_outcomes):
            p = marginal(belief, t, o)
            if p:
                q += p * stop_value(update(belief, t, o))
        if q > best:
            best, act = q, ai
    return act


def greedy_policy(problem: Problem):
    def pi(belief, budget):
        return greedy_action(problem, belief, budget)

    return pi


def policy_value(problem: Problem, policy_fn, belief=None, budget=None) -> Fraction:
    """Exact expected value of a memoryless policy via memoized recursion."""
    belief = problem.prior if belief is None else belief
    budget = problem.budget if budget is None else budget
    memo = {}

    def rec(b, B):
        key = (b, B)
        if key in memo:
            return memo[key]
        a = policy_fn(b, B)
        if a is STOP:
            v = stop_value(b)
        else:
            t = problem.tools[a]
            v = -problem.kappa * t.cost
            for o in range(t.n_outcomes):
                p = marginal(b, t, o)
                if p:
                    v += p * rec(update(b, t, o), B - t.cost)
        memo[key] = v
        return v

    return rec(belief, budget)
