"""Exact finite-budget planner: Bellman recursion over (belief, budget).

V(b, B) = max( max_z b(z),
               max_{a : c_a <= B} [ -kappa * c_a
                   + sum_{o : P(o|b,a) > 0} P(o|b,a) * V(b^{a,o}, B - c_a) ] )

Recursion terminates because every tool cost is a positive integer, so the
budget strictly decreases along every call edge (depth <= initial budget).
"""
from __future__ import annotations

from fractions import Fraction

from .belief import Belief, best_guess, outcome_prob, stop_value, update
from .model import Problem

STOP = "stop"
CALL = "call"

# An action is ("stop", guess_index) or ("call", tool_index).
Action = tuple


class ExactPlanner:
    """Optimal planner with exact rational arithmetic.

    Cache keys are the exact belief tuples (Fractions) plus the remaining
    budget. Two distinct beliefs are never merged, so caching cannot change
    the induced policy: the reference path does no rounding at all.
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.cache: dict[tuple[Belief, int], tuple[Fraction, Action]] = {}

    def _key(self, belief: Belief, budget: int):
        return (tuple(belief), budget)

    def value(self, belief: Belief, budget: int) -> Fraction:
        key = self._key(belief, budget)
        hit = self.cache.get(key)
        if hit is not None:
            return hit[0]
        best_value = stop_value(belief)
        best_action: Action = (STOP, best_guess(belief))
        for tool_index, tool in enumerate(self.problem.tools):
            if tool.cost > budget:
                continue  # budget may never go negative
            q = -self.problem.kappa * tool.cost
            for o in range(tool.n_outcomes):
                p_o = outcome_prob(belief, tool, o)
                if p_o == 0:
                    continue  # zero-probability branch: no contribution
                posterior = update(belief, tool, o)
                q += p_o * self.value(posterior, budget - tool.cost)
            if q > best_value:
                best_value = q
                best_action = (CALL, tool_index)
        self.cache[key] = (best_value, best_action)
        return best_value

    def action(self, belief: Belief, budget: int) -> Action:
        self.value(belief, budget)
        return self.cache[self._key(belief, budget)][1]

    def policy(self):
        """Memoryless policy callable: (belief, budget) -> action.

        The hidden state z is not and cannot be an argument.
        """
        return self.action


class FloatPlanner(ExactPlanner):
    """Approximate planner whose cache key rounds beliefs to `ndigits` floats.

    Distinct beliefs whose rounded float forms coincide share one cache
    entry, so the returned action can depend on insertion order and may
    differ from the exact planner. This class exists to demonstrate that
    risk; the exact reference path is ExactPlanner.
    """

    def __init__(self, problem: Problem, ndigits: int = 12):
        super().__init__(problem)
        self.ndigits = ndigits

    def _key(self, belief: Belief, budget: int):
        return (tuple(round(float(b), self.ndigits) for b in belief), budget)
