"""Finite-budget information-acquisition agent with exact Bayesian planning.

All value and belief arithmetic uses fractions.Fraction so that every
reported number is exactly recomputable from the inputs.
"""
from .model import Problem, Tool, problem_from_dict, problem_to_dict
from .belief import best_guess, marginal, stop_value, update
from .planner import STOP, OptimalPlanner, RoundedCachePlanner
from .baselines import (
    FixedOrderPolicy,
    fixed_order_value,
    greedy_action,
    greedy_policy,
    immediate_stop_value,
    policy_value,
)

__all__ = [
    "Problem",
    "Tool",
    "problem_from_dict",
    "problem_to_dict",
    "best_guess",
    "marginal",
    "stop_value",
    "update",
    "STOP",
    "OptimalPlanner",
    "RoundedCachePlanner",
    "FixedOrderPolicy",
    "fixed_order_value",
    "greedy_action",
    "greedy_policy",
    "immediate_stop_value",
    "policy_value",
]
