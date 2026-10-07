"""Portable finite selector regression; no campaigns, saved data, or timings.

The independent reference enumerates vertices of the four-category completion
polytope split at the candidate coordinates. It does not use accepts,
completion_distance, the simulator, or a copied historical implementation.
"""
from __future__ import annotations

from fractions import Fraction as F
from itertools import product
from math import sqrt
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from learning import AuditTape, AuditView, select


def dictionary():
    roots = ((1/16, 7/16, 7/16, 1/16), (1/4,)*4,
             (7/16, 1/16, 1/16, 7/16))
    children = ((3/8, 3/8, 1/8, 1/8), (1/4,)*4,
                (1/8, 1/8, 3/8, 3/8))
    return list(product(roots, children, children))


def literal_view(marks, returned):
    visible = set(returned)
    histogram = tuple(sum(marks[j] == z for j in visible) for z in range(4))
    prefix = next((j for j in range(len(marks)) if j not in visible), len(marks))
    prefix_histogram = tuple(marks[:prefix].count(z) for z in range(4))
    return AuditView(len(marks), prefix, histogram, prefix_histogram)


def records():
    result = []
    for seed in range(36):
        views = []
        for i in range(3):
            n = (seed + 3*i) % 9
            marks = tuple((j*j + seed + i) % 4 for j in range(n))
            if seed % 4 == 0:
                returned = range(n)
            elif seed % 4 == 1:
                returned = range(1, n)  # one old blocker
            elif seed % 4 == 2:
                returned = [j for j in range(n) if marks[j] == 0]
            else:
                returned = []
            views.append(literal_view(marks, returned))
        result.append(views)
    return result


def completion_vertex_distance(view, law):
    if not view.n:
        return 0.0
    lower = tuple(F(x, view.n) for x in view.histogram)
    target = tuple(F(x) for x in law)
    distances = []
    # A linear TV cell has a minimum at a vertex. Three coordinates there
    # are either a lower-bound facet or a TV kink; normalization fixes the
    # fourth. Enumeration is finite and separate from the producer formula.
    for free in range(4):
        fixed = [z for z in range(4) if z != free]
        for values in product(*[(lower[z], target[z]) for z in fixed]):
            point = [F(0)]*4
            for z, value in zip(fixed, values):
                point[z] = value
            point[free] = 1 - sum(values)
            if all(point[z] >= lower[z] for z in range(4)):
                distances.append(sum(abs(point[z]-target[z]) for z in range(4))/2)
    if not distances:
        raise ValueError("reference fixture has no completion")
    return float(min(distances))


def literal_membership(view, law, method, c):
    if method in ("prefix", "intersection") and view.prefix:
        frequencies = [view.prefix_histogram[z]/view.prefix for z in range(4)]
        distance = sum(abs(p-q) for p, q in zip(law, frequencies))/2
        if distance > min(1.0, sqrt(c/view.prefix)) + 1e-12:
            return False
    if method == "intersection" and view.n:
        if completion_vertex_distance(view, law) > min(1.0, sqrt(c/view.n)) + 1e-12:
            return False
    if method == "completed" and sum(view.histogram):
        size = sum(view.histogram)
        frequencies = [x/size for x in view.histogram]
        distance = sum(abs(p-q) for p, q in zip(law, frequencies))/2
        if distance > min(1.0, sqrt(c/size)) + 1e-12:
            return False
    return True


def literal_select(method, views, laws, values, policies, c):
    admitted = []
    for candidate, triple in enumerate(laws):
        decisions = [literal_membership(views[i], triple[i], method, c) for i in range(3)]
        if False not in decisions:
            admitted.append(candidate)
    if not admitted:
        return 0, (), True
    chosen = sorted((values[j], j) for j in admitted)[0][1]
    return policies[chosen], tuple(admitted), False


class SelectorMembershipTests(unittest.TestCase):
    def test_independent_literal_records(self):
        laws = dictionary()
        policies = list(range(100, 127))
        for views in records():
            for method, c, values in product(
                ("prefix", "intersection", "completed"), (0.0, 0.25, 6.9),
                (list(range(27)), [1.0]*27),
            ):
                self.assertEqual(select(method, views, laws, values, policies, c),
                                 literal_select(method, views, laws, values, policies, c))

    def test_duplicates_reordering_type_keys_and_fallback(self):
        atom0, atom1 = (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)
        views = [literal_view((z,)*4, range(4)) for z in (0, 1, 0)]
        good, bad = (atom0, atom1, atom0), (atom0, atom0, atom0)
        for laws in ([bad, good, good], [good, bad, good], [bad, bad], []):
            for method in ("prefix", "intersection", "completed"):
                policies = list(range(40, 40+len(laws)))
                values = [0]*len(laws)
                self.assertEqual(select(method, views, laws, values, policies, 0),
                                 literal_select(method, views, laws, values, policies, 0))

    def test_lazy_first_encounters_and_call_local_reset(self):
        laws = dictionary()
        empty = [AuditView(0, 0, (0,)*4, (0,)*4) for _ in range(3)]
        original = AuditView.accepts
        calls = []

        def counted(view, law, method, c):
            calls.append((next(i for i, v in enumerate(empty) if v is view), law))
            return original(view, law, method, c)

        expected = []
        for row in laws:
            for i, law in enumerate(row):
                if (i, law) not in expected:
                    expected.append((i, law))
        with patch.object(AuditView, "accepts", counted):
            for c in (0.0, 1.0):
                start = len(calls)
                result = select("intersection", empty, laws, [0]*27, list(range(27)), c)
                self.assertEqual(result, (0, tuple(range(27)), False))
                self.assertEqual(calls[start:], expected)
        self.assertEqual(len(expected), 9)
        # An early rejection still leaves a malformed later type unvisited.
        view = literal_view((0,), (0,))
        malformed = [(tuple([0.0, 1.0, 0.0, 0.0]),)]
        self.assertEqual(select("prefix", [view], malformed, [], [], 0), (0, (), True))

    def test_mutable_and_custom_objects_keep_predicate_calls(self):
        atom = (1.0, 0.0, 0.0, 0.0)
        view = AuditView(0, 0, (0,)*4, (0,)*4)
        original = AuditView.accepts
        for laws, views in (
            ([(list(atom),)*3]*2, [view]*3),
            ([(atom,)*3]*2, [AuditView(0, 0, [0]*4, [0]*4)]*3),
            ([(tuple(F(x) for x in atom),)*3]*2, [view]*3),
        ):
            calls = []

            def counted(v, law, method, c):
                calls.append((v, law))
                return original(v, law, method, c)

            with patch.object(AuditView, "accepts", counted):
                self.assertEqual(select("prefix", views, laws, [0, 0], [4, 5], 0),
                                 (4, (0, 1), False))
            self.assertEqual(len(calls), 6)

        class ChangingView:
            def __init__(self):
                self.calls = 0

            def accepts(self, law, method, c):
                self.calls += 1
                law[0] += 1  # admitted custom predicate with mutable input
                return self.calls == 1

        changing = ChangingView()
        law = [1.0, 0.0, 0.0, 0.0]
        result = select("prefix", [changing]*3, [(law,)*3]*2, [0, 0], [4, 5], 0)
        self.assertEqual(result, (0, (), True))
        self.assertEqual((changing.calls, law[0]), (3, 4.0))

        views = [literal_view((0,), (0,)), None, view]

        class ReplacingView:
            def accepts(self, law, method, c):
                views[0] = literal_view((1,), (0,))
                return True

        views[1] = ReplacingView()
        self.assertEqual(select("prefix", views, [(atom,)*3]*2, [0, 0], [4, 5], 0),
                         (4, (0,), False))

    def test_slack_special_paths_and_error_order(self):
        view = literal_view((0,), (0,))
        for offset in (0.0, 0.5e-12, 2.0e-12, -2.0e-12):
            law = (0.75-offset, 0.25+offset, 0.0, 0.0)
            for method in ("prefix", "intersection", "completed"):
                laws = [(law,)*3]*2
                self.assertEqual(select(method, [view]*3, laws, [0, 0], [6, 7], .0625),
                                 literal_select(method, [view]*3, laws, [0, 0], [6, 7], .0625))
        self.assertEqual(select("safe", None, None, None, None, None), (0, None, False))
        empty = [AuditView(0, 0, (0,)*4, (0,)*4)]*3
        self.assertEqual(select("plugin", empty, dictionary(), [], list(range(27)), -1),
                         (13, None, False))
        with self.assertRaisesRegex(ValueError, "Unknown learning method: invalid"):
            select("invalid", None, None, None, None, None)
        with self.assertRaises(AttributeError):
            select("prefix", [None], [()], [], [], 0)
        with self.assertRaises(IndexError):
            select("prefix", [view]*3, [((),)*3], [], [], 0)
        with self.assertRaisesRegex(ValueError, "math domain error"):
            select("prefix", [view]*3, [((1.0, 0.0, 0.0, 0.0),)*3], [0], [0], -1)
        # Empty records have no active arithmetic/shape check, even for c<0.
        self.assertEqual(select("intersection", empty, [((),)*3], [0], [5], -1),
                         (5, (0,), False))

    def test_snapshot_payload_isolation_and_new_boundary(self):
        left, right = [AuditTape() for _ in range(3)], [AuditTape() for _ in range(3)]
        for a, b in zip(left, right):
            a.launch(0, 9)
            b.launch(3, 9)
        for method in ("prefix", "intersection", "completed", "plugin", "safe"):
            a, b = [t.snapshot() for t in left], [t.snapshot() for t in right]
            self.assertEqual(a, b)
            self.assertTrue(all(not hasattr(v, "ready") and not hasattr(v, "marks") for v in a+b))
            self.assertEqual(select(method, a, dictionary(), [0]*27, list(range(27)), 0),
                             select(method, b, dictionary(), [0]*27, list(range(27)), 0))
        for tape in left:
            tape.receive(9)
        views = [t.snapshot() for t in left]
        for method in ("prefix", "intersection", "completed"):
            self.assertEqual(select(method, views, dictionary(), [0]*27, list(range(27)), 0),
                             literal_select(method, views, dictionary(), [0]*27, list(range(27)), 0))


if __name__ == "__main__":
    unittest.main()
