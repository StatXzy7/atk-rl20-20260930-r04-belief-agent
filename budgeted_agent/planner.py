"""Exact finite-budget Bellman planner over (belief, remaining_budget).

V(b, B) = max( V_stop(b), max over tools a with cost <= B of Q(b, B, a) )
Q(b, B, a) = -kappa * cost_a + sum_o P(o|b,a) * V(b^{a,o}, B - cost_a)
V_stop(b) = max_z b(z)

Costs are positive integers, so each recursive call strictly decreases the
budget and the recursion terminates after at most B levels. Zero-probability
observations contribute nothing and are skipped. Ties are broken
deterministically: STOP first, then lowest tool index.

The cache key is the exact rational belief, so no two distinct beliefs are
ever merged. RoundedCachePlanner is provided only as an UNSOUND contrast:
it rounds beliefs before keying, which can merge distinct beliefs and flip
the optimal action (see tests/test_rounding_trap.py).
"""
from __future__ import annotations

from fractions import Fraction

from .belief import marginal, stop_value, update
from .model import Problem

STOP = None  # action encoding: None means "stop and report best_guess(belief)"


class OptimalPlanner:
    def __init__(self, problem: Problem):
        self.problem = problem
        self._cache = {}
        self.expansions = 0

    def _key(self, belief, budget):
        return (belief, budget)

    def value(self, belief=None, budget=None) -> Fraction:
        belief = self.problem.prior if belief is None else belief
        budget = self.problem.budget if budget is None else budget
        key = self._key(belief, budget)
        hit = self._cache.get(key)
        if hit is not None:
            return hit[0]
        self.expansions += 1
        best_value = stop_value(belief)
        best_action = STOP
        for ai, tool in enumerate(self.problem.tools):
            if tool.cost > budget:
                continue
            q = -self.problem.kappa * tool.cost
            for o in range(tool.n_outcomes):
                p_o = marginal(belief, tool, o)
                if p_o == 0:
                    continue
                q += p_o * self.value(update(belief, tool, o), budget - tool.cost)
            if q > best_value:
                best_value, best_action = q, ai
        self._cache[key] = (best_value, best_action)
        return best_value

    def policy(self, belief=None, budget=None):
        belief = self.problem.prior if belief is None else belief
        budget = self.problem.budget if budget is None else budget
        self.value(belief, budget)
        return self._cache[self._key(belief, budget)][1]

    def stats(self) -> dict:
        return {"cache_size": len(self._cache), "expansions": self.expansions}


class RoundedCachePlanner(OptimalPlanner):
    """UNSOUND variant: rounds belief components to ndigits before keying.

    Demonstrates why approximate cache merging is dangerous: two beliefs that
    round to the same key share one cached (value, action) pair, so the
    returned action can be suboptimal for the actual belief. The exact
    reference planner never rounds.
    """

    def __init__(self, problem: Problem, ndigits: int = 2):
        super().__init__(problem)
        self.ndigits = ndigits

    def _key(self, belief, budget):
        return (tuple(round(float(x), self.ndigits) for x in belief), budget)
