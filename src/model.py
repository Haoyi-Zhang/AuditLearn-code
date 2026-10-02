"""A finite, synthetic one-fork speculative scheduling model.

There is no model inference, GPU use, network access, or imported baseline.
All scheduling semantics are in evaluate_world and described in README.md.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from typing import Iterable

ROOT_MARKS = ((0, 1), (1, 1), (0, 2), (1, 2))
CHILD_MARKS = ((0, 1), (0, 2), (1, 1), (1, 2))
# Policy action at observations (early root 0, early root 1, root pending).
# Actions 0=wait, 1=launch child 0, 2=launch child 1.
POLICIES = tuple(product(range(3), repeat=3))
WASTE_PRICE = F(1, 4)

@dataclass(frozen=True)
class Model:
    correlation: F
    success0: F
    success1: F

    @property
    def laws(self) -> tuple[tuple[F, ...], ...]:
        r = self.correlation
        root = ((1+r)/4, (1-r)/4, (1-r)/4, (1+r)/4)
        def child(p: F) -> tuple[F, ...]:
            return ((1-p)/2, (1-p)/2, p/2, p/2)
        return root, child(self.success0), child(self.success1)


def models() -> tuple[Model, ...]:
    return tuple(Model(r, a, b) for r, a, b in product(
        (F(-3,4), F(0), F(3,4)), (F(1,4), F(1,2), F(3,4)),
        (F(1,4), F(1,2), F(3,4))))


def evaluate_world(policy: tuple[int, int, int], world: tuple[int, int, int]) -> tuple[F, int, tuple[int, ...], int]:
    """Return centered cost, wasted launches, launched type ids, finish time.

    Root starts at time zero. At time one, process arrivals, then choose a
    single optional launch. A root verification starts a deterministic
    two-tick fallback. Only a strictly earlier successful correct child is
    useful; a tie counts as wasted work. All launched marks are later audited.
    """
    branch, root_delay = ROOT_MARKS[world[0]]
    observation = branch if root_delay == 1 else 2
    action = policy[observation]
    finish = root_delay + 2
    waste = 0
    launched = (0,)
    if action:
        child = action - 1
        success, child_delay = CHILD_MARKS[world[action]]
        candidate = max(root_delay, 1 + child_delay)
        useful = bool(child == branch and success and candidate < finish)
        if useful:
            finish = candidate
        else:
            waste = 1
        launched = (0, action)
    residual = F(finish - root_delay) + WASTE_PRICE*waste
    return residual, waste, launched, finish


def exact_value(model: Model, policy: tuple[int, int, int]) -> tuple[F, F, tuple[F, F, F]]:
    """World enumeration; this is one of two separately written oracles."""
    cost, waste = F(0), F(0)
    launches = [F(0), F(0), F(0)]
    laws = model.laws
    for world in product(range(4), repeat=3):
        prob = laws[0][world[0]] * laws[1][world[1]] * laws[2][world[2]]
        c, w, active, _ = evaluate_world(policy, world)
        cost += prob*c
        waste += prob*w
        for i in active:
            launches[i] += prob
    return cost, waste, tuple(launches)


def analytic_oracle(model: Model) -> tuple[F, tuple[int, int, int]]:
    """Independent backward calculation over the three decision observations.

    No call to evaluate_world, exact_value, or policy enumeration. The formulas
    below are independently derived from the conditional branch probabilities.
    """
    r = model.correlation
    observation_mass = ((1+r)/4, (1-r)/4, F(1,2))
    pending_branch = ((1-r)/2, (1+r)/2)
    chosen, value = [], F(0)
    for observation in range(3):
        qvalues = [F(2)]
        for child, p in enumerate((model.success0, model.success1)):
            if observation < 2:
                # Only a one-tick successful child beats the early fallback.
                useful_prob = p/2 if child == observation else F(0)
                saving = useful_prob
            else:
                useful_prob = pending_branch[child]*p
                saving = F(3,2)*useful_prob
            qvalues.append(F(2) - saving + WASTE_PRICE*(1-useful_prob))
        action = min(range(3), key=lambda a: (qvalues[a], a))
        chosen.append(action)
        value += observation_mass[observation]*qvalues[action]
    return value, tuple(chosen)


def tv(p: Iterable[F | float], q: Iterable[F | float]):
    return sum(abs(a-b) for a, b in zip(p, q))/2


def draw_index(probabilities: tuple[float, ...], u: float) -> int:
    acc = 0.0
    for j, p in enumerate(probabilities):
        acc += p
        if u < acc:
            return j
    return len(probabilities)-1
