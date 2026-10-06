"""Finite exact checks, not a machine-checked general theorem."""
from __future__ import annotations
from fractions import Fraction as F
from itertools import product, permutations
from math import sqrt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from model import models, POLICIES, exact_value, analytic_oracle, tv, ROOT_MARKS, evaluate_world
from learning import AuditTape, AuditView, select, confidence_constant


def compositions(n, d):
    if d == 1:
        yield (n,)
    else:
        for first in range(n+1):
            for tail in compositions(n-first, d-1):
                yield (first,)+tail


def check():
    all_models = models()
    values, launches, wastes = [], [], []
    matches = 0
    action_values_checked = 0
    for law in all_models:
        triplets = [exact_value(law, p) for p in POLICIES]
        row = [x[0] for x in triplets]
        best = min(range(len(POLICIES)), key=lambda j: (row[j], j))
        independent_value, independent_policy = analytic_oracle(law)
        assert (row[best], POLICIES[best]) == (independent_value, independent_policy)
        assert all(F(0) <= x <= F(9,4) for x in row)
        # Post-pilot audit: independently check every policy, not only minima.
        masses = ((1+law.correlation)/4, (1-law.correlation)/4, F(1,2))
        late_branch = ((1-law.correlation)/2, (1+law.correlation)/2)
        for policy, actual in zip(POLICIES, triplets):
            value, waste = F(0), F(0)
            counts = [F(1), F(0), F(0)]
            for obs, action in enumerate(policy):
                cost, wasted = F(2), F(0)
                if action:
                    child = action-1
                    success = (law.success0,law.success1)[child]
                    useful = (success/2 if child==obs else F(0)) if obs<2 else late_branch[child]*success
                    saved = useful if obs<2 else F(3,2)*useful
                    wasted = 1-useful
                    cost = 2-saved+F(1,4)*wasted
                    counts[action] += masses[obs]
                value += masses[obs]*cost
                waste += masses[obs]*wasted
            assert actual == (value,waste,tuple(counts))
            action_values_checked += 1
        values.append(row)
        wastes.append([x[1] for x in triplets])
        launches.append([x[2] for x in triplets])
        matches += 1
    # Unlaunched child marks must not affect any reported episode result.
    noninterference_checks = 0
    for policy in POLICIES:
        for world in product(range(4), repeat=3):
            actual = evaluate_world(policy, world)
            assert F(0) <= actual[0] <= F(9,4)
            assert 1 <= len(actual[2]) <= 2
            for child in (1,2):
                if child not in actual[2]:
                    for replacement in range(4):
                        changed = list(world); changed[child] = replacement
                        assert evaluate_world(policy, tuple(changed)) == actual
                        noninterference_checks += 1
    # Every ordering of four distinct audit arrivals preserves the prefix.
    audit_order_checks = 0
    for order in permutations(range(4)):
        tape = AuditTape()
        for mark in range(4):
            tape.launch(mark, order.index(mark)+1)
        seen = set()
        for time,mark in enumerate(order,1):
            tape.receive(time); seen.add(mark)
            prefix = 0
            while prefix in seen: prefix += 1
            assert tape.prefix == prefix
            assert tape.histogram == [int(z in seen) for z in range(4)]
            assert tape.prefix_histogram == [int(z<prefix) for z in range(4)]
            audit_order_checks += 1
    # Different pending payloads give equal immutable observation-only views.
    # The selection signature has no true-model argument.
    policies = [min(range(27), key=lambda j:(values[i][j],j)) for i in range(27)]
    minima = [float(values[i][policies[i]]) for i in range(27)]
    laws = [[tuple(map(float,x)) for x in m.laws] for m in all_models]
    hidden_a,hidden_b = [AuditTape() for _ in range(3)],[AuditTape() for _ in range(3)]
    for a,b in zip(hidden_a,hidden_b):
        a.launch(0,100); b.launch(3,100)
    selection_information_checks = 0
    for method in ('prefix','intersection','completed','plugin','safe'):
        a_views, b_views = [t.snapshot() for t in hidden_a], [t.snapshot() for t in hidden_b]
        assert all(not hasattr(t,"ready") and not hasattr(t,"marks") for t in a_views+b_views)
        left = select(method,a_views,laws,minima,policies,confidence_constant(512,.05))
        right = select(method,b_views,laws,minima,policies,confidence_constant(512,.05))
        assert left == right
        selection_information_checks += 1
    # Exhaustive pairwise checks of the fixed-policy launch-weighted coupling bound.
    coupling_checks = 0
    max_ratio = F(0)
    for a, b in product(range(27), repeat=2):
        distances = [tv(all_models[a].laws[i], all_models[b].laws[i]) for i in range(3)]
        for policy in range(27):
            left = abs(values[a][policy]-values[b][policy])
            right = F(9,4)*sum(launches[a][policy][i]*distances[i] for i in range(3))
            assert left <= right
            if right:
                max_ratio = max(max_ratio, left/right)
            coupling_checks += 1
    # Exact distance-to-completion-set calculation on a rational lattice.
    # q and possible completed histograms share denominator n. This is an
    # exhaustive finite grid, not validation of every real-valued simplex point.
    completion_checks = 0
    for n in range(1, 6):
        full = list(compositions(n, 3))
        for observed in (c for k in range(n+1) for c in compositions(k, 3)):
            possible = [e for e in full if all(e[z] >= observed[z] for z in range(3))]
            assert possible
            for q in full:
                brute = min(sum(abs(e[z]-q[z]) for z in range(3)) for e in possible)/F(2*n)
                formula = sum(max(0, observed[z]-q[z]) for z in range(3))/F(n)
                assert brute == formula
                completion_checks += 1
    # Cross-check the implemented four-category completion distance against
    # brute-force integer completions on every rational grid through n=4.
    implementation_completion_distance_checks = 0
    for n in range(1, 5):
        full = list(compositions(n, 4))
        for observed in (c for k in range(n + 1) for c in compositions(k, 4)):
            possible = [e for e in full if all(e[z] >= observed[z] for z in range(4))]
            view = AuditView(n, 0, observed, (0, 0, 0, 0))
            for q in full:
                brute = min(
                    sum(abs(e[z] - q[z]) for z in range(4)) / F(2 * n)
                    for e in possible
                )
                implemented = view.completion_distance(tuple(float(x / n) for x in q))
                assert abs(implemented - float(brute)) < 1e-12
                implementation_completion_distance_checks += 1

    # The integer completion set has the same exact TV diameter u/n as the
    # fractional polytope whenever at least two categories are available.
    # This checks that fractional relaxation does not create the ambiguity term.
    completion_diameter_checks = 0
    for n in range(1, 7):
        full = list(compositions(n, 3))
        for observed in (c for k in range(n + 1) for c in compositions(k, 3)):
            possible = [e for e in full if all(e[z] >= observed[z] for z in range(3))]
            brute = max(
                sum(abs(a[z] - b[z]) for z in range(3)) / F(2 * n)
                for a in possible for b in possible
            )
            unresolved = n - sum(observed)
            assert brute == F(unresolved, n)
            completion_diameter_checks += 1

    # Exact saturated outstanding-record envelope for unit batches.  If at most
    # U reports are missing before any launch, the largest possible cumulative
    # unresolved fraction is sum_{s=1}^{n-1} min(U,s)/s.
    saturated_envelope_checks = 0
    for U in range(17):
        for n in range(0, 513):
            direct = sum((F(min(U, s), s) for s in range(1, n)), F(0))
            if n <= 1 or U == 0:
                closed = F(0)
            elif U >= n - 1:
                closed = F(n - 1)
            else:
                closed = F(U) + U * sum((F(1, s) for s in range(U + 1, n)), F(0))
            assert direct == closed
            saturated_envelope_checks += 1
    envelope_boundary_regressions = {
        "n0": sum((F(min(4, s), s) for s in range(1, 0)), F(0)),
        "n1": sum((F(min(4, s), s) for s in range(1, 1)), F(0)),
        "u0_n9": sum((F(min(0, s), s) for s in range(1, 9)), F(0)),
        "u_ge_n": sum((F(min(12, s), s) for s in range(1, 9)), F(0)),
    }
    assert envelope_boundary_regressions == {
        "n0": F(0), "n1": F(0), "u0_n9": F(0), "u_ge_n": F(8)
    }
    single_blocker_n = 256
    single_blocker_ambiguity = sum((F(1, s) for s in range(1, single_blocker_n)), F(0))
    assert single_blocker_ambiguity == sum(
        (F(min(1, s), s) for s in range(1, single_blocker_n)), F(0)
    )

    # Known-radius binary arithmetic regression.  This is not an off-grid
    # campaign: it checks the corrected width algebra exactly.
    center = F(1, 2)
    radius = F(1, 100)
    cover_radius = F(1, 10)
    cover_point = F(3, 5)
    selected_candidate = F(39, 100)
    assert abs(cover_point - center) <= radius + cover_radius
    assert abs(selected_candidate - center) <= radius + cover_radius
    assert abs(center - selected_candidate) > 2 * radius
    assert abs(center - selected_candidate) <= 2 * radius + cover_radius
    assert abs(cover_point - selected_candidate) <= 2 * (radius + cover_radius)
    known_radius_binary_regression = {
        "center": str(center),
        "radius": str(radius),
        "cover_radius": str(cover_radius),
        "cover_point": str(cover_point),
        "selected_candidate": str(selected_candidate),
        "true_to_candidate": str(abs(center - selected_candidate)),
        "candidate_to_cover": str(abs(cover_point - selected_candidate)),
        "corrected_direct_width": str(2 * radius + cover_radius),
        "full_inflated_diameter": str(2 * (radius + cover_radius)),
    }

    # Worst valid prefix length at every batch; all x_t in {0,1,2}, T=8.
    counting_checks = 0
    for k in (0, 1, 2, 7):
        for sequence in product(range(3), repeat=8):
            n, lhs = 0, 0.0
            for x in sequence:
                m = max(0, n-k)
                w = 1.0 if not m else min(1.0, 2*sqrt(1/m))
                lhs += x*w
                n += x
            rhs = k+2+4*sqrt(n)
            assert lhs <= rhs+1e-12
            counting_checks += 1
    # Completion-width counting bound.  At each boundary, at most U reports
    # remain outstanding, even though an old missing report can make prefix
    # debt much larger.  The worst width for fixed N is attained at u=min(U,N).
    completion_counting_checks = 0
    for U in (0, 1, 2, 7):
        for sequence in product(range(3), repeat=8):
            n, lhs, unresolved_mass = 0, 0.0, 0.0
            for x in sequence:
                if n == 0:
                    width = 1.0
                else:
                    u = min(U, n)
                    width = min(1.0, 2*sqrt(1/n)+u/n)
                    unresolved_mass += x*u/n
                lhs += x*width
                n += x
            # Check the actual completion-exposure lemma before its coarser
            # outstanding-count corollary. All launches in a batch share N.
            assert lhs <= 2+4*sqrt(n)+unresolved_mass+1e-12
            harmonic = sum(1/k for k in range(1, max(0, n-2)+1))
            rhs = 2+4*sqrt(n)+U*harmonic
            assert lhs <= rhs+1e-12
            completion_counting_checks += 1
    # Identical root marginals, different optimal pending action.
    a, b = 8, 26  # both children have success probability 3/4
    assert all_models[a].success0 == all_models[b].success0 == F(3,4)
    def margins(model):
        p = model.laws[0]
        return (p[0]+p[2], p[1]+p[3]), (p[0]+p[1], p[2]+p[3])
    assert margins(all_models[a]) == margins(all_models[b])
    pa = analytic_oracle(all_models[a])[1]
    pb = analytic_oracle(all_models[b])[1]
    assert pa[2] == 1 and pb[2] == 2
    product_policy = analytic_oracle(all_models[17])[1]
    product_id = POLICIES.index(product_policy)
    positive_regret = values[b][product_id] - min(values[b])
    assert positive_regret > 0
    # One pending oldest report can hold back 256 prefix positions. Treating
    # pending count as prefix debt would falsify the numerical counting bound.
    n = 256
    wrong_debt_rhs = 1+1+4*sqrt(n)
    assert n > wrong_debt_rhs
    # Selected completed feedback is not a random sample of the full tape.
    tape = AuditTape()
    for j in range(80):
        tape.launch(0 if j % 2 == 0 else 3, 1 if j % 2 == 0 else 100)
    tape.receive(1)
    assert tape.histogram == [40, 0, 0, 0]
    assert tape.prefix == 1
    assert tape.n-tape.prefix == 79 and tape.n-tape.observed == 40
    # Correct joint confidence contains this exactly balanced underlying tape;
    # the completed-only confidence rejects at a deterministic radius 0.2.
    law = (F(1,2), F(0), F(0), F(1,2))
    distance_complete = sum(max(F(0), F(tape.histogram[z], tape.n)-law[z]) for z in range(4))
    distance_naive = tv(law, [F(x,tape.observed) for x in tape.histogram])
    assert distance_complete == 0 and distance_naive == F(1,2)
    return {
        "oracle_matches": matches,
        "independent_policy_triplets": action_values_checked,
        "unlaunched_mark_noninterference_checks": noninterference_checks,
        "audit_order_checks": audit_order_checks,
        "selection_information_checks": selection_information_checks,
        "policy_values_exact": len(all_models)*len(POLICIES),
        "worlds_per_policy_value": 64,
        "coupling_checks": coupling_checks,
        "maximum_coupling_ratio": str(max_ratio),
        "completion_lattice_checks": completion_checks,
        "implementation_completion_distance_checks": implementation_completion_distance_checks,
        "completion_diameter_lattice_checks": completion_diameter_checks,
        "saturated_outstanding_envelope_checks": saturated_envelope_checks,
        "outstanding_envelope_boundary_regressions": {
            k: str(v) for k, v in envelope_boundary_regressions.items()
        },
        "known_radius_binary_regression": known_radius_binary_regression,
        "single_blocker_exact_unresolved_mass_exposure": {
            "launches": single_blocker_n,
            "identity": "H_255",
            "decimal": float(single_blocker_ambiguity),
        },
        "prefix_counting_checks": counting_checks,
        "completion_counting_checks": completion_counting_checks,
        "same_marginals_policies": {"negative_correlation": pa, "positive_correlation": pb},
        "product_policy": product_policy,
        "product_oracle_regret_per_episode_positive_correlation": str(positive_regret),
        "prefix_debt_counterexample": {"launches": n, "maximum_unreturned": 1, "wrong_bound": wrong_debt_rhs},
        "selective_return_counterexample": {"launched": 80, "observed": 40, "prefix": 1,
            "true_success_fraction": "1/2", "completed_success_fraction": "0", "completion_set_distance": "0"},
        "status": "all finite checks passed; general proofs are separate mathematical arguments"
    }, values, wastes

if __name__ == "__main__":
    import json
    result, _, _ = check()
    print(json.dumps(result, indent=2))
