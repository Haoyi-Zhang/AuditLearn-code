# Known-radius dictionary approximation and conditional audit blackout

This supplement records two deliberately narrow boundary results. The first
extends the finite-dictionary theorem when a valid typewise cover radius is
known in advance. The second is a conditional indistinguishability statement;
the current finite fork does not instantiate its premise.

## Model domain and sensitivity

Let

`Q = product_i Delta(Z_i)`

be the full product-law domain on which every causal policy in `Pi` and the
bounded centered episode cost are defined. The planning dictionary `Theta` is
a finite subset of `Q`. For every type `i`, at most `b_i` marks can be launched
in one episode and the absolute centered episode cost is at most `L`.

Use the explicit conservative sensitivity

`B_i := L b_i`.

The launch-coupling lemma then gives, for any policy `pi` and any `q,q' in Q`
that agree outside coordinate `i`,

`|V_q(pi)-V_q'(pi)| <= B_i TV(q_i,q'_i)`.

The definition and bound are over the full model domain `Q`, not just pairs of
dictionary points.

## Proposition 1: inflated coverage for a known radius

Suppose the true law `p in Q` need not belong to `Theta`, but a declared vector
`epsilon_i in [0,1]` and a cover point `p_bar in Theta` satisfy

`TV(p_i,p_bar_i) <= epsilon_i` for every type `i`.

Inflate the type-`i` prefix test from radius `r_i(M_it)` to
`r_i(M_it)+epsilon_i`, and the completion-distance test from
`r_i(N_it)` to `r_i(N_it)+epsilon_i`. On the simultaneous tape event from the
main theorem, `p_bar` is feasible for both inflated tests.

For the prefix test this is the triangle inequality:

`TV(p_bar_i,p_hat_i,M) <= epsilon_i + r_i(M_it)`.

For completion confidence, distance to a nonempty set is 1-Lipschitz:

`dist_TV(p_bar_i,E_it) <= epsilon_i + dist_TV(p_i,E_it)`
`                           <= epsilon_i + r_i(N_it)`.

Feasibility of `p_bar` does **not** allow the nominal `2r` width to be reused.

## Proposition 2: true-to-selected-candidate width

Let `q_t` be any candidate selected from the inflated intersection set. Direct
comparison with the true law gives

`TV(p_i,q_t,i) <= w_tilde_pre_it`

with

`w_tilde_pre_it := min{1, 2 r_i(M_it) + epsilon_i}`,

and

`TV(p_i,q_t,i) <= w_tilde_cmp_it`

with

`w_tilde_cmp_it := min{1, 2 r_i(N_it)`
`                         + u_it/max{1,N_it} + epsilon_i}`.

For the prefix route, one nominal radius joins `p_i` to the prefix empirical
law and one inflated radius joins that empirical law to `q_t,i`. For the
completion route, join `p_i` to one feasible completion, cross the completion
set using its exact diameter `u_it/N_it`, and join another feasible completion
to `q_t,i` using the inflated radius. Hence

`w_tilde_cap_it := min{w_tilde_pre_it,w_tilde_cmp_it}`
`                <= w_cap_it + epsilon_i`.

This is the required correction to the uninflated-width argument.

## Proposition 3: regret with known cover radii

Let

`mu^p_it := E_p[X_it | Hist_t]`

be the predictable launch occupancy under the **true** environment. The
coupling from `p` to the selected candidate `q_t` must use this occupancy. An
occupancy computed under `p_bar` is not the learner's actual launch occupancy.

On the good event, optimism compares the selected pair `(q_t,pi_t)` with the
feasible pair `(p_bar,pi_p^*)`. Coupling the selected policy from `p` to `q_t`
and telescoping the oracle policy from `p_bar` to `p` gives

`V_p(pi_t)-V_p(pi_p^*)`
` <= L sum_i mu^p_it w_tilde_cap_it`
`    + planning_error_t + sum_i B_i epsilon_i`
` <= L sum_i mu^p_it w_cap_it`
`    + L sum_i mu^p_it epsilon_i`
`    + planning_error_t + sum_i B_i epsilon_i`.

The widths are measurable at the decision boundary, so the same tower-property
step as in the main theorem gives

`E_p[mu^p_it w_cap_it] = E_p[X_it w_cap_it]`

without conditioning on the future-dependent global confidence event. Summing
and handling the failure event separately yields

`R_T(p)`
` <= L E_p[sum_t,i X_it w_cap_it]`
`    + L sum_i epsilon_i E_p[N_i,T+1]`
`    + T sum_i B_i epsilon_i`
`    + sum_t planning_error_t + L T delta`.

Since `N_i,T+1 <= T b_i` and `B_i=L b_i`, a convenient conservative form is

`R_T(p)`
` <= (nominal actual-exposure bound)`
`    + 2 T sum_i B_i epsilon_i`.

Thus the compact approximation penalty remains valid, but only after charging
`epsilon_i` in the true-to-selected-candidate width and using the true-law
predictable occupancy.

## Binary arithmetic regression

Take a binary law with true and empirical success probability `0.50`, nominal
radius `r=0.01`, known cover radius `epsilon=0.10`, cover point `0.60`, and
selected candidate `0.39`. Both dictionary candidates pass the inflated radius
`r+epsilon=0.11` around `0.50`.

However,

- the nominal diameter `2r=0.02` does not bound
  `TV(0.50,0.39)=0.11`;
- it also does not bound the candidate-to-cover distance
  `TV(0.60,0.39)=0.21`;
- the corrected direct width `2r+epsilon=0.12` does bound the true-to-selected
  distance; and
- the full inflated-set diameter `2(r+epsilon)=0.22` bounds the distance
  between the two feasible dictionary points.

`python run.py check` verifies these rational equalities and inequalities. It is
an exact arithmetic regression, not an off-grid experiment.

## What the artifact does and does not contain

The released artifact contains exactly three reported empirical campaigns:
`pilot`, `stress`, and `scale`. It has no dictionary-external input file, no
inflated-confidence selector, and no off-grid result. Therefore the
known-radius result above is reported only as a proved theoretical extension.
No dictionary-internal run is renamed as misspecification evidence, and no
adaptive estimate of an unknown cover radius is claimed.

## Conditional audit-blackout necessity

For this statement, visible history includes the learner's chosen policies and
actions, all physical arrivals, nonarrivals and within-episode outcomes, the
realized scalar centered cost when observed, the learner's own randomization,
and all audits returned so far.

Assume two admissible environments have the same law for this **entire** visible
history during the first `h` episodes in which no new audit is returned. Assume
also that their unique optimal policies are opposite and that selecting the
wrong policy has gap at least `Delta` in each environment. Then every randomized
learner has expected regret at least `h Delta/2` in one of the two environments
over the blackout interval.

The proof is the standard two-environment averaging argument. Equality in law
of the visible history forces the same conditional policy distribution in both
environments. If `alpha_t` is the probability of choosing the policy optimal
only in environment 1, the two one-step regrets average at least

`Delta [alpha_t + (1-alpha_t)] / 2 = Delta/2`.

Summing over `h` episodes and taking the larger environment-specific regret
proves the conditional statement.

The current finite fork generally reveals physical events and scalar costs that
can differ across environments. The artifact does not provide a pair satisfying
full-history indistinguishability. Consequently this supplement does not claim
an unconditional blackout lower bound for the fork, nor a matching lower bound
for prefix debt, unresolved-mass exposure, or their minimum.
