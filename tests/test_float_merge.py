"""Caching risk: rounding-based cache keys can merge distinct beliefs and
thereby change the induced policy. The exact planner never rounds."""
from fractions import Fraction as F

from budgeted_info_agent import ExactPlanner, FloatPlanner
from budgeted_info_agent.examples import bsc_example

EPS = F(1, 10**13)
HALF = F(1, 2)
BELIEF_ARGMAX_0 = (HALF + EPS, HALF - EPS)
BELIEF_ARGMAX_1 = (HALF - EPS, HALF + EPS)


def test_exact_planner_distinguishes_close_beliefs():
    prob = bsc_example(budget=0)  # only stopping is available
    planner = ExactPlanner(prob)
    assert planner.action(BELIEF_ARGMAX_0, 0) == ("stop", 0)
    assert planner.action(BELIEF_ARGMAX_1, 0) == ("stop", 1)


def test_float_planner_merges_and_changes_policy():
    prob = bsc_example(budget=0)
    planner = FloatPlanner(prob, ndigits=12)
    first = planner.action(BELIEF_ARGMAX_0, 0)
    second = planner.action(BELIEF_ARGMAX_1, 0)
    # both beliefs round to (0.5, 0.5): the second query reuses the first
    # cache entry and reports the wrong argmax
    assert first == ("stop", 0)
    assert second == ("stop", 0)
    assert len(planner.cache) == 1


def test_exact_planner_never_merges_distinct_beliefs():
    prob = bsc_example(budget=0)
    planner = ExactPlanner(prob)
    planner.action(BELIEF_ARGMAX_0, 0)
    planner.action(BELIEF_ARGMAX_1, 0)
    assert len(planner.cache) == 2
