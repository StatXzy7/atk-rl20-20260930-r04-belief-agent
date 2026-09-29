"""Independent verification by explicit enumeration.

These routines deliberately avoid the Bellman/memo structure of planner.py:
values are computed by summing over explicit root-to-leaf observation paths,
and optimality is checked by brute-force maximization over ALL deterministic
memoryless policies on the reachable state DAG (feasible only for tiny trees).
"""
from __future__ import annotations

import itertools
from fractions import Fraction

from .belief import marginal, stop_value, update
from .model import Problem
from .planner import STOP


def reachable_states(problem: Problem, belief=None, budget=None):
    """All (belief, budget) states reachable with positive probability."""
    start = (
        problem.prior if belief is None else belief,
        problem.budget if budget is None else budget,
    )
    seen = set()
    stack = [start]
    while stack:
        b, B = stack.pop()
        if (b, B) in seen:
            continue
        seen.add((b, B))
        for t in problem.tools:
            if t.cost > B:
                continue
            for o in range(t.n_outcomes):
                if marginal(b, t, o) > 0:
                    stack.append((update(b, t, o), B - t.cost))
    return seen


def available_actions(problem: Problem, belief, budget):
    return [STOP] + [ai for ai, t in enumerate(problem.tools) if t.cost <= budget]


def _assignment_value(problem: Problem, assign, start) -> Fraction:
    memo = {}

    def rec(state):
        if state in memo:
            return memo[state]
        b, B = state
        a = assign[state]
        if a is STOP:
            v = stop_value(b)
        else:
            t = problem.tools[a]
            v = -problem.kappa * t.cost
            for o in range(t.n_outcomes):
                p = marginal(b, t, o)
                if p:
                    v += p * rec((update(b, t, o), B - t.cost))
        memo[state] = v
        return v

    return rec(start)


def enumerate_all_policies(problem: Problem):
    """Brute-force max over every deterministic memoryless policy.

    Returns (best_value, best_assignment, n_policies). Only usable on very
    small observation trees: the count is the product of per-state action
    counts.
    """
    start = (problem.prior, problem.budget)
    states = sorted(reachable_states(problem), key=repr)
    action_lists = [available_actions(problem, b, B) for b, B in states]
    n_policies = 1
    for acts in action_lists:
        n_policies *= len(acts)
    best_value = None
    best_assign = None
    for combo in itertools.product(*action_lists):
        assign = dict(zip(states, combo))
        v = _assignment_value(problem, assign, start)
        if best_value is None or v > best_value:
            best_value, best_assign = v, assign
    return best_value, best_assign, n_policies


def policy_value_by_paths(problem: Problem, policy_fn, belief=None, budget=None):
    """Exact expected value of a memoryless policy, computed by enumerating
    every root-to-leaf path of the policy's observation tree and summing
    path_probability * (expected correctness - kappa * spent_cost).

    Returns (value, n_paths). Raises if path probabilities do not sum to 1.
    """
    start_b = problem.prior if belief is None else belief
    start_B = problem.budget if budget is None else budget
    total = Fraction(0)
    prob_sum = Fraction(0)
    n_paths = 0
    stack = [(start_b, start_B, Fraction(1), Fraction(0))]
    while stack:
        b, B, p, spent = stack.pop()
        a = policy_fn(b, B)
        if a is STOP:
            total += p * (stop_value(b) - problem.kappa * spent)
            prob_sum += p
            n_paths += 1
            continue
        t = problem.tools[a]
        for o in range(t.n_outcomes):
            p_o = marginal(b, t, o)
            if p_o:
                stack.append((update(b, t, o), B - t.cost, p * p_o, spent + t.cost))
    if prob_sum != 1:
        raise AssertionError(f"policy tree probabilities sum to {prob_sum}, not 1")
    return total, n_paths


def count_tree_paths(problem: Problem, belief=None, budget=None) -> int:
    """Number of root-to-leaf paths in the FULL observation tree (every tool
    tried at every node). Grows like (sum_a |O_a|)^depth; contrast with the
    number of distinct cached (belief, budget) states."""
    b = problem.prior if belief is None else belief
    B = problem.budget if budget is None else budget
    leaves = 1  # stopping here is one leaf
    for t in problem.tools:
        if t.cost > B:
            continue
        for o in range(t.n_outcomes):
            if marginal(b, t, o) > 0:
                leaves += count_tree_paths(problem, update(b, t, o), B - t.cost)
    return leaves
