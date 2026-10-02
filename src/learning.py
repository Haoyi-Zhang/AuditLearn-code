"""Transparent finite-dictionary optimistic learning with delayed audit marks."""
from __future__ import annotations
from dataclasses import dataclass, field
from math import log, sqrt

@dataclass
class AuditTape:
    marks: list[int | None] = field(default_factory=list)
    ready: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    histogram: list[int] = field(default_factory=lambda: [0]*4)
    prefix_histogram: list[int] = field(default_factory=lambda: [0]*4)
    prefix: int = 0

    def launch(self, mark: int, due_episode: int) -> None:
        index = len(self.marks)
        self.marks.append(None)
        self.ready.setdefault(due_episode, []).append((index, mark))

    def receive(self, episode: int) -> None:
        # Reports are reliable and have exact episode keys; no silently lost
        # tasks or index relabeling. Only snapshot() views reach the learner.
        for index, mark in self.ready.pop(episode, []):
            assert self.marks[index] is None
            self.marks[index] = mark
            self.histogram[mark] += 1
        while self.prefix < len(self.marks) and self.marks[self.prefix] is not None:
            mark = self.marks[self.prefix]
            assert mark is not None
            self.prefix_histogram[mark] += 1
            self.prefix += 1

    @property
    def n(self) -> int:
        return len(self.marks)

    @property
    def observed(self) -> int:
        return sum(self.histogram)

    def snapshot(self) -> "AuditView":
        """Return an immutable observation-only view, excluding pending payloads."""
        return AuditView(self.n, self.prefix, tuple(self.histogram), tuple(self.prefix_histogram))


@dataclass(frozen=True)
class AuditView:
    n: int
    prefix: int
    histogram: tuple[int, ...]
    prefix_histogram: tuple[int, ...]

    @property
    def observed(self) -> int:
        return sum(self.histogram)

    def radius(self, n: int, c: float) -> float:
        return 1.0 if n == 0 else min(1.0, sqrt(c/n))

    def prefix_width(self, c: float) -> float:
        """Diameter bound from the fully returned chronological prefix."""
        return min(1.0, 2.0*self.radius(self.prefix, c))

    def completion_width(self, c: float) -> float:
        """Diameter bound from all completions of the returned histogram."""
        if self.n == 0:
            return 1.0
        missing = self.n-self.observed
        return min(1.0, 2.0*self.radius(self.n, c)+missing/self.n)

    def intersection_width(self, c: float) -> float:
        return min(self.prefix_width(c), self.completion_width(c))

    def completion_distance(self, law: tuple[float, ...]) -> float:
        """TV distance from ``law`` to the feasible completion polytope.

        Returned category counts impose coordinate-wise lower bounds
        ``histogram[z] / n``.  For a probability vector, the total excess over
        these lower bounds is exactly the distance to that polytope.
        """
        if self.n == 0:
            return 0.0
        return sum(
            max(0.0, self.histogram[z] / self.n - law[z])
            for z in range(len(self.histogram))
        )

    def accepts(self, law: tuple[float, ...], method: str, c: float) -> bool:
        # c is the theorem's simultaneous finite-horizon radius constant.
        if method in ("prefix", "intersection") and self.prefix:
            dist = sum(abs(law[z] - self.prefix_histogram[z]/self.prefix) for z in range(4))/2
            if dist > self.radius(self.prefix, c)+1e-12:
                return False
        if method == "intersection" and self.n:
            distance = self.completion_distance(law)
            if distance > self.radius(self.n, c)+1e-12:
                return False
        if method == "completed" and self.observed:
            # Intentional negative-control estimator. Return order can be biased.
            dist = sum(abs(law[z] - self.histogram[z]/self.observed) for z in range(4))/2
            if dist > self.radius(self.observed, c)+1e-12:
                return False
        return True


def select(method: str, tapes: list[AuditView], laws, optimal_values, optimal_policies,
           c: float) -> tuple[int, tuple[int, ...] | None, bool]:
    """Select using public model candidates and immutable observed statistics only.

    Returns policy id, feasible candidate ids (when applicable), and fallback.
    The simulator computes true-law exclusion outside this function. Known-law
    and product-law reference policies are also handled outside this learner.
    """
    if method == "safe":
        return 0, None, False
    if method == "plugin":
        estimates = [tuple((t.histogram[z]+1)/(t.observed+4) for z in range(4)) for t in tapes]
        closest = min(range(len(laws)), key=lambda k: (
            sum(sum(abs(laws[k][i][z]-estimates[i][z]) for z in range(4)) for i in range(3)), k))
        return optimal_policies[closest], None, False
    if method not in ("prefix", "intersection", "completed"):
        raise ValueError(f"Unknown learning method: {method}")
    feasible = tuple(k for k in range(len(laws)) if all(
        tapes[i].accepts(laws[k][i], method, c) for i in range(3)))
    if not feasible:
        return 0, feasible, True
    k = min(feasible, key=lambda j: (optimal_values[j], j))
    return optimal_policies[k], feasible, False


def confidence_constant(horizon: int, delta: float) -> float:
    if horizon < 1 or not 0 < delta < 1:
        raise ValueError("horizon must be positive and delta in (0,1)")
    # m=3 types, B=2 launches/episode, d=4 categorical marks.
    return (4*log(2)+log(3*2*horizon/delta))/2
