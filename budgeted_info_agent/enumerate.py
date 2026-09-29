"""Independent verification: explicit tree unrolling and brute-force
enumeration over all adaptive policies. Shares no code with the planner's
memoized recursion, so agreement is a genuine cross-check.
"""
from __future__ import annotations

import itertools
from fractions import Fraction

from .belief import Belief, outcome_prob, stop_value, update
from .model import Problem


def unrolled_optimal_value(
    problem: Problem,
    belief: Belief | None = None,
    budget: int | None = None,
    counter: list | None = None,
) -> Fraction:
    """Bellman optimum recomputed by expanding the full observation tree
    with no memoization. `counter[0]` counts expanded nodes (growth check).
    Only feasible for small budgets."""
    if belief is None:
        belief, budget = problem.prior, problem.budget
    if counter is not None:
        counter[0] += 1
    best = stop_value(belief)
    for tool in problem.tools:
        if tool.cost > budget:
            continue
        q = -problem.kappa * tool.cost
        for o in range(tool.n_outcomes):
            p_o = outcome_prob(belief, tool, o)
            if p_o == 0:
                continue
            q += p_o * unrolled_optimal_value(
                problem, update(belief, tool, o), budget - tool.cost, counter
            )
        if q > best:
            best = q
    return best


def _build_history_tree(problem: Problem, max_nodes: int):
    """Full tree of reachable histories. Nodes are NOT merged: two histories
    with identical posteriors remain distinct nodes, so the policy class
    enumerated below is the full adaptive (history-dependent) class."""
    decision_nodes = []

    def build(belief: Belief, budget: int) -> dict:
        node = {"belief": belief, "budget": budget, "children": {}}
        affordable = [
            ti for ti, t in enumerate(problem.tools) if t.cost <= budget
        ]
        if affordable:
            decision_nodes.append(node)
            if len(decision_nodes) > max_nodes:
                raise ValueError("tree too large for brute-force enumeration")
            for ti in affordable:
                tool = problem.tools[ti]
                branches = []
                for o in range(tool.n_outcomes):
                    p_o = outcome_prob(belief, tool, o)
                    if p_o == 0:
                        continue
                    branches.append(
                        (p_o, build(update(belief, tool, o), budget - tool.cost))
                    )
                node["children"][ti] = branches
        return node

    root = build(problem.prior, problem.budget)
    return root, decision_nodes


def brute_force_optimal(problem: Problem, max_nodes: int = 100000):
    """Max expected payoff over ALL adaptive policies, found by enumerating
    every assignment of actions to every reachable history and scoring each
    policy by exact path enumeration. Returns (best_value, n_policies)."""
    root, decision_nodes = _build_history_tree(problem, max_nodes)
    options = [["stop"] + sorted(node["children"]) for node in decision_nodes]

    def score(assignment) -> Fraction:
        choice = {
            id(node): act for node, act in zip(decision_nodes, assignment)
        }

        def value(node) -> Fraction:
            act = choice.get(id(node), "stop")
            if act == "stop":
                return stop_value(node["belief"])
            tool = problem.tools[act]
            return -problem.kappa * tool.cost + sum(
                p * value(child) for p, child in node["children"][act]
            )

        return value(root)

    best_value = None
    count = 0
    for assignment in itertools.product(*options):
        v = score(assignment)
        count += 1
        if best_value is None or v > best_value:
            best_value = v
    return best_value, count
