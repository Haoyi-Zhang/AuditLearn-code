# Learning speculative schedules from selectively delayed audits

This document is the standalone mathematical supplement for the paper.  The
arguments below are hand proofs, not a formal mechanization.  The executable
checks in `tests/check_exact.py` validate only the finite cases reported in
`results/exact_checks.json`; they do not prove the general statements.

The result concerns a finite-model learner with reliable audits of every
launched task.  It does **not** claim a general solution to censored feedback,
a practical decoder, a hardware speedup, or minimax-optimal delay dependence.

## 1. Model, two clocks, and comparator

Fix a horizon `T`, task types `i=1,...,m`, and a finite categorical alphabet
`Z_i` of size `d_i>=2` for each type.  The unknown parameter
`p=(p_1,...,p_m)` lies in a known finite dictionary `Theta`, and the truth is in
`Theta`.

Each type owns an infinite independent tape

    Z_i,1, Z_i,2, ...  iid~ p_i.

A launch of type `i` consumes the next unused mark before its value is known.
A mark may jointly encode an outcome, a physical duration, and any other finite
state needed to score the episode.  Coordinates of the same mark may be dependent; different type tapes are
independent.

An episode chooses a causal policy `pi` from a finite class `Pi`.  A policy may
react to physical arrivals, nonarrivals, and pending states inside the episode.
There are at most `B` launches per episode and at most `b_i` launches of type
`i`.  Once the launched marks and parameter-independent common random coins are
fixed, the episode and its cost are fixed.

Let `C` be completion time plus a nonnegative price on wasted launches.  A
known policy-independent baseline `A`, determined only by mandatory marks and
common coins, centers the objective so that

    ell = C - A  in [0,L]

for every policy, model, and possible episode.  Define

    V_p(pi) = E_p[ell(pi)],
    pi_p^* in argmin_{pi in Pi} V_p(pi).

The comparator knows `p`, but not future marks.  It receives the same physical
observations, has the same launch limits, and pays the same cost as the learner.
Expected pseudo-regret is

    R_T = E sum_{t=1}^T [V_p(pi_t)-V_p(pi_p^*)].

Physical verification and learning telemetry use different clocks.  Every
launch has a unique audit record that eventually reveals its whole mark and its
chronological launch index.  Audit time may depend on the mark and reports may
arrive out of order.  Audits update the learner only at episode boundaries and
do not alter physical dynamics in the episode that produced the mark.

Reporting is nonanticipating: timing and payloads can depend on already
launched marks, past history, and independent reporting coins, but not on
unconsumed tape entries. Hence, conditional on the history before a launch,
the next unused type-`i` mark still has law `p_i`. This fresh-mark condition is
needed when fixed-policy launch occupancy is used conditionally in the regret
proof; the simultaneous confidence event alone does not imply it.

Before episode `t`, let

- `N_it` be the number of earlier type-`i` launches;
- `c_it(z)` be the number of returned type-`i` marks equal to `z`;
- `O_it=sum_z c_it(z)` be the number of returned audits;
- `u_it=N_it-O_it` be the number still unreturned; and
- `M_it` be the length of the fully returned chronological prefix.

The chronological prefix debt is `N_it-M_it`.  It can be much larger than
`u_it`: one old missing record can block the prefix while later records return.
We use two delay descriptions:

    N_it - M_it <= K_i                                            (1)

and the realized launch-normalized unresolved-mass exposure

    Gamma_i^miss(T) = sum_{t=1}^T X_it u_it / max{1,N_it},          (2)

where `X_it` is the number of type-`i` launches in episode `t`.  The learner
need not know `K_i` or future unresolved-mass exposure.

## 2. A simultaneous concentration event on launch tapes

For `delta in (0,1)`, define

    a_i = [d_i log 2 + log(mBT/delta)]/2,
    r_i(0)=1,
    r_i(n)=min{1,sqrt(a_i/n)}  for n>=1.                           (3)

Let `p_hat_i,n` be the empirical distribution of the first `n` marks on the
*launch tape*, including marks whose audits have not yet returned.

**Lemma 1 (uniform launch-prefix event).** With probability at least
`1-delta`, simultaneously for every type `i` and every `1<=n<=BT`,

    TV(p_i,p_hat_i,n) <= r_i(n).                                  (4)

**Proof.**  For a fixed subset `S` of the alphabet, the indicators
`1{Z_i,j in S}` are Bernoulli.  Hoeffding's moment-generating-function argument
gives

    P[p_hat_i,n(S)-p_i(S)>x] <= exp(-2 n x^2).

For finite categorical laws,

    TV(q,p) = max_{S subseteq Z_i} [q(S)-p(S)].

A union bound over the `2^{d_i}` subsets, then over the `mBT` type-prefix
pairs, gives (4).  When `r_i(n)=1`, violation is impossible.  QED.

The event is defined for *all* deterministic launch-tape prefixes before a
random index is selected.  It therefore remains valid at the adaptive indices
`M_it` and `N_it`.  This does not say that the subset of reports that happened
to return is iid.

## 3. Exact completion geometry for selected returns

For `N=N_it>0`, define the fractional completion polytope

    E_it = {e in simplex(Z_i): e(z) >= c_it(z)/N for every z}.      (5)

For `N=0`, take the whole simplex. The latent full empirical histogram
`p_hat_i,N` belongs to this set on every sample path. Also define the feasible
integer empirical completions

    E_it^int = {(c_it+h)/N: h in N^{d_i}, sum_z h(z)=u_it}.          (6)

**Lemma 2 (exact completion geometry).** For `N>0` and any law `q`,

    dist_TV(q,E_it) = sum_z (c_it(z)/N - q(z))_+,                  (7)
    diam_TV(E_it) = diam_TV(E_it^int) = u_it/N.                    (8)

Consequently, any set constructed from this record alone that is guaranteed to
contain every feasible integer completion has TV diameter at least `u_it/N`.

**Proof.** Write `v_z=c_it(z)/N` and `h=sum_z(v_z-q_z)_+`. Every feasible point
must increase `q` by total mass at least `h` on deficient coordinates, so its
TV distance is at least `h`. Conversely, `max(v,q)` has total mass `1+h`. At
least `h` mass can be removed from coordinates above their lower bounds without
crossing those bounds, giving a feasible unit vector at TV distance exactly
`h`.

Every fractional completion has the form

    v + (u_it/N)b

for a simplex vector `b`, so the fractional diameter is at most `u_it/N`. If
`u_it>0`, assign all missing counts to category `a` or to a distinct category
`b`. These are integer completions at TV distance exactly `u_it/N`; when
`u_it=0`, both sets are singletons. The final containment statement follows
from the same two witnesses. QED.

Thus the fractional relaxation does not inflate the worst-case ambiguity. The
denominator must be total launches `N_it`, not returned audits `O_it`.
Normalizing by `O_it` would silently treat the selected returned subset as the
whole sample.

## 4. Prefix, completion, and intersection confidence sets

Define the chronological-prefix constraint

    P_it = {q_i: TV(q_i,p_hat_i,M_it) <= r_i(M_it)},               (9)

using a vacuous constraint when `M_it=0`, and the completion constraint

    H_it = {q_i: dist_TV(q_i,E_it) <= r_i(N_it)}.                  (10)

The joint confidence set is

    C_t = Theta intersect product_i (P_it intersect H_it).        (11)

On the event of Lemma 1, the true law lies in every `C_t`: the witness for
`H_it` is the latent full empirical histogram `p_hat_i,N_it`.

Define

    w_it^pre = min{1, 2 r_i(M_it)},                                (12)
    w_it^cmp = min{1, 2 r_i(N_it)+u_it/max{1,N_it}},               (13)
    w_it^cap = min{w_it^pre,w_it^cmp}.                             (14)

**Lemma 3 (usable law width).** If `p,q in C_t`, then

    TV(p_i,q_i) <= w_it^cap                                       (15)

for every type `i`.

**Proof.** The prefix inequality is the triangle inequality around
`p_hat_i,M_it`. For completion, choose closest points in the compact polytope
`E_it` to `p_i` and `q_i`. Each law is within `r_i(N_it)` of its point, and
Lemma 2 puts the two points within `u_it/N_it`. Cap at one when `N_it=0` and
take the smaller valid bound. QED.

At a fixed boundary the intersection is no wider than either constituent.
This pointwise fact does not by itself imply lower runwise regret, because the
chosen policy changes future launches and therefore future audit records.

## 5. Launch-sensitive value coupling

**Lemma 4 (coupling at actual launches).** For a fixed causal policy `pi` and
any `p,q in Theta`,

    |V_p(pi)-V_q(pi)|
      <= L sum_i E_p[X_i(pi)] TV(p_i,q_i).                         (15)

**Proof.**  Couple each fresh pair of type-`i` marks maximally when the common
history reaches that launch.  Before the first mark disagreement, the two
physical histories and decisions are identical.  The probability that a given
launch is the first disagreement is at most the probability the `p` execution
reaches that launch times `TV(p_i,q_i)`.  Summing over launch positions gives
the disagreement probability on the right without `L`.  Centered costs agree
when no mark disagrees and differ by at most `L` otherwise.  QED.

The coupling uses whole joint marks; it does not divide by the probability of
a rare pending state.  It is an occupancy-weighted smoothness argument, not a
claim that such couplings are new in general.

## 6. Optimism and two pathwise exposure descriptions

At boundary `t`, if `C_t` is nonempty, choose `q_t in C_t` and `pi_t in Pi`
satisfying

    V_qt(pi_t) <= min_{q in C_t, pi in Pi} V_q(pi) + epsilon_t,    (16)

where `epsilon_t>=0` is deterministic.  If the set is empty, execute a fixed
safe policy and record the fallback.  The finite implementation uses exact
precomputed rational policy values, zero planning tolerance, and floating-point
membership tests with a `1e-12` acceptance slack.  The theorem is for the ideal
real-arithmetic rule.

**Lemma 5 (two exposure descriptions).** For every type `i`, pathwise,

    sum_t X_it w_it^pre
      <= K_i + b_i + 4 sqrt(a_i N_i,T+1),                          (17)

    sum_t X_it w_it^cmp
      <= b_i + 4 sqrt(a_i N_i,T+1) + Gamma_i^miss(T),               (18)

and

    sum_t X_it w_it^cap
      <= b_i + 4 sqrt(a_i N_i,T+1)
         + min{K_i,Gamma_i^miss(T)}.                                (19)

**Proof.**  Index the type-`i` launches by `j=1,...,N_i,T+1`.  If launch `j`
occurs in episode `t`, then `N_it>=j-b_i` because at most `b_i` same-type
launches share the episode.

For prefix confidence, (1) gives `M_it>=j-b_i-K_i`.  Charge one to the first
`K_i+b_i` launches.  Every later launch is charged at most
`2 sqrt(a_i/(j-b_i-K_i))`.  Since

    sum_{k=1}^n k^{-1/2} <= 2 sqrt(n),

(17) follows.

For completion confidence, charge one to the first `b_i` launches.  Every later
launch is charged at most

    2 sqrt(a_i/(j-b_i)) + u_it/max{1,N_it}.

The first terms sum to at most `4 sqrt(a_i N_i,T+1)`; the second terms are
exactly (2), giving (18).  Since `w^cap` is pointwise no larger than either
component, its cumulative sum is no larger than the minimum of (17) and (18).
The common part factors out, yielding (19).  QED.

The shared-square-root structure is important: taking a pointwise minimum and
then separately upper-bounding two unrelated totals would not generally
produce the minimum of two delay penalties.  Here the two bounds have the same
estimation term by construction.

## 7. Dual-descriptor regret theorem

**Theorem 6 (audited finite-model learning).** Under the preceding assumptions,

    R_T <= L E[sum_t sum_i X_it w_it^cap]
           + sum_t epsilon_t + LT delta,                           (20)

and hence

    R_T <= L sum_i [b_i + 4 sqrt(a_i E N_i,T+1)
                    + E min{K_i,Gamma_i^miss(T)}]
           + sum_t epsilon_t + LT delta,                           (21)

and

    R_T <= L [sum_i (b_i+E min{K_i,Gamma_i^miss(T)})
              +4 sqrt(BT sum_i a_i)]
           + sum_t epsilon_t + LT delta.                           (22)

Every right-hand side may be replaced by its minimum with the trivial bound
`LT`.  Prefix-only optimism satisfies (20) with `w^pre` and (17).

**Proof.**  Let `G={p in C_t for every t}`.  On `G`, the true comparator is
feasible in (16) at every boundary.  Lemmas 3 and 4 give, conditional on the
boundary history,

    V_p(pi_t)-V_p(pi_p^*)
      <= L sum_i mu_it w_it^cap + epsilon_t,

where `mu_it=E[X_it | history]`.  The per-round gap is always at most `L`.
Insert the indicator of `G` into the valid inequality, drop it from the
nonnegative exposure term, and bound the complement by
`LT P(G^c)<=LT delta`.  Each width is boundary-measurable, so

    E[mu_it w_it^cap] = E[X_it w_it^cap]

by the tower property.  Thus (20) never conditions that identity on a
future-dependent event.

Apply (19) pathwise, take expectations, and use Jensen's inequality for the
square root to obtain (21).  Finally,
`sum_i E N_i,T+1 <= BT` and Cauchy--Schwarz give

    sum_i sqrt(a_i E N_i,T+1) <= sqrt(BT sum_i a_i),

which proves (22).  QED.

The theorem keeps the realized minimum of chronological debt and launch-normalized unresolved-mass exposure.  It does not require the learner to know either descriptor,
and it does not assert minimax optimality.

## 8. Sharp outstanding-record envelope

For integers `U,n>=0`, define

    Psi_U(n) = sum_{s=1}^{(n-1)_+} min{U,s}/s.                      (24)

**Corollary 7 (saturated unit-batch envelope).** Suppose `b_i=1` and
`u_it <= U_i` at every boundary. Define `H_0=0` and

`Psi_U(n) = sum_{s=1}^{(n-1)_+} min{U,s}/s`,

with an empty sum equal to zero. Then pathwise

`Gamma_i^miss(T) <= Psi_U_i(N_i,T+1)`.

The exact closed form, including zero launches, is

- `Psi_U(n)=0` when `n<=1` or `U=0`;
- `Psi_U(n)=n-1` when `n>=2` and `U>=n-1`; and
- `Psi_U(n)=U+U(H_{n-1}-H_U)` when `n>=2` and `1<=U<n-1`.

At the `j`th launch, the denominator is `j-1` and the outstanding count is at
most `min{U,j-1}`. Summing gives the formula. A front-blocker schedule attains
equality term by term. For one blocker the exact descriptor is
`H_{(N_i,T+1-1)_+}`. An unlaunched type contributes zero. The logarithmic
upper bound is interpreted with `log max{1,N_i,T+1}`.

## 9. A separate exploration--waste inequality

This proposition concerns a two-action subproblem and is not a matching lower
bound for Theorem 6.

A safe action costs `F`.  A probe succeeds with probability `p`, costs `F-s`
on success and `F+lambda` on failure, and only probing reveals a Bernoulli
mark.  A failure contributes one unit of raw waste.  Set

    p0 = lambda/(s+lambda),
    p_- = p0-eta,
    p_+ = p0+eta,
    g = (s+lambda)eta,

with `0<eta<min{p0,1-p0}`.  Safe is optimal under `p_-` and probe under `p_+`.
Let `N` be the number of probes, `W` their failures, and
`h=kl(p_-,p_+)`.

**Proposition 8 (learning under a waste budget).** If `E_-[W]<=w` and
`b=w/(1-p_-)`, then

    R_+(T) >= g [T-b-T sqrt(bh/2)]_+.                              (24)

**Proof.**  Predictable probe choices and fresh Bernoulli marks give
`E_-[W]=(1-p_-)E_-[N]`, hence `E_-[N]<=b`.  Augment the transcript by all probe
marks, even if their reports have not arrived.  Policy kernels and the common
conditional reporting kernels cancel in the likelihood ratio.  Chain rule for
KL gives at most `E_-[N]h`; projecting back to the observed transcript cannot
increase KL.  Pinsker and `0<=N<=T` imply

    E_+[N] <= E_-[N] + T sqrt(E_-[N]h/2)
            <= b + T sqrt(bh/2).

The plus-world regret is `g(T-E_+[N])`; take the positive part.  QED.

This is a specialization of standard adaptive change-of-measure reasoning.  It
shows that an exogenous low-waste requirement can suppress the observations
needed in a nearby profitable environment; it is not a delay-specific lower
bound.

## 10. Exact finite fork and counterexamples

The executable model uses a mandatory root with mark `(Y,D)`, where
`Y in {0,1}`, `D in {1,2}`, and probabilities

    ((1+rho)/4,(1-rho)/4,(1-rho)/4,(1+rho)/4)

in order `((0,1),(1,1),(0,2),(1,2))`.  At physical time one, arrivals are
processed before choosing to wait, launch child 0, or launch child 1.  A root
arrival starts a two-tick fallback.  Child `j` succeeds with probability `p_j`
and has duration uniform on `{1,2}`.  It is useful only if it is successful,
on the selected branch, and strictly earlier than fallback.  Cost is completion
time plus `1/4` per wasted optional launch; centering by root duration gives
`ell in [0,9/4]`.

There are three observations and three actions, hence `3^3=27` deterministic
policies.  The dictionary uses

    rho in {-3/4,0,3/4},
    p_0,p_1 in {1/4,1/2,3/4},

for 27 models.  All 64 triples of marks are enumerated for every model-policy
pair.  A separately written three-node backward oracle agrees with direct
world enumeration on all 729 value/waste/launch triplets and all 27 minima.

Three exact counterexamples delimit the claims.

1. **Joint-law necessity.**  With `p_0=p_1=3/4`, the optimal pending action
   differs between `rho=-3/4` and `rho=3/4` even though root marginals agree.
   The fixed product-marginal policy incurs regret `63/128` per episode under
   the positive-correlation law.
2. **Selected returns are not an empirical prefix.**  On an alternating
   length-80 tape, return all zero marks and withhold all one marks.  The
   returned success fraction is zero, the full empirical fraction is one half,
   the completed prefix has length one, and the completion set still contains
   one half.
3. **Missing count is not prefix debt.**  Withhold the first audit across 256
   later decisions while returning all later records promptly.  At most one
   audit is unreturned, yet prefix debt reaches 256; replacing `K_i` by the
   missing count would falsely bound a width sum of 256 by 66.

## 11. Executable evidence

`python run.py check` currently records:

- 27 independent oracle-minimum matches;
- 729 exact policy value/waste/launch triplets;
- 19,683 launch-coupling inequalities;
- 1,973 completion-polytope projection cases;
- 209 exact fractional/integer diameter cases;
- 26,244 prefix-exposure counting cases;
- 26,244 completion-exposure counting cases;
- 8,721 saturated outstanding-envelope cases;
- 9,216 unlaunched-mark noninterference cases;
- 96 audit-order states; and
- five selector information-boundary checks.

The frozen diagnostic pilot contains 252 runs and 129,024 episodes.  The
completed-only negative control excludes the truth in 213 dependent episode
records; the valid prefix and intersection rules have no exclusions in this
pilot.  The plug-in baseline is often better, and that unfavorable result is
retained.

The all-dictionary stress campaign contains 1,512 runs and 774,144 episodes,
covering all 27 truths, four seeds, seven methods, and two audit scenarios.  In
the single-blocker scenario, the maximum number of unreturned audits is one and
the maximum prefix debt is 511.  Across 108 truth-seed cells, intersection
confidence has lower pseudo-regret than prefix confidence in 76 cells, ties in
27, and is worse in five; its mean difference is `-65.0811`.  These are paired
descriptive results on the exact finite model, not a population estimate.

The prespecified horizon/backlog campaign adds 600 runs and 476,160 episodes
across horizons 128--2,048, three truths, four seeds, five methods, and front
blocker counts one and four. All 600 rows attain the exact envelope
`sum_i Psi_U(N_i)`. Across its ten `(U,T)` slices, intersection is never worse
than prefix in any of the 12 paired truth--seed cells. This sensitivity result
is descriptive and does not establish an asymptotic rate.

## 12. Scope and remaining uncertainty

The proof requires iid independent type tapes, a true-containing finite
model dictionary, bounded launches and centered cost, a finite policy class,
and reliable full audits of every launched mark.  It excludes arbitrary
misspecification, task-tape dependence, destroyed/censored marks, anonymous
reports, adversarially corrupted audits, and an efficient oracle for general
trees.

The completion-polytope idea is related to prior missing-outcome confidence
intervals, and the launch-sensitive coupling is related to triggered-bandit
smoothness. Recent capacity-constrained delayed optimization also derives a
harmonic backlog cost under strong convexity. The paper therefore does not
claim novelty for harmonic weighting or for charging outstanding observations.
Its narrower claimed delta is the exact fractional-and-integer categorical
completion geometry for identified full audits, the launch-normalized exposure
forced by that geometry, and its transfer through a causal occupancy coupling
alongside the chronological-prefix alternative. Confidence-set optimism,
categorical concentration, and occupancy coupling are not individually new.

The finite checks and campaigns validate implementation consistency and expose
failure modes; they do not establish deployment performance or replace
independent peer review.  The explicit finite-horizon bound can exceed the
trivial `LT` bound at horizon 512, and this negative calibration is reported.
