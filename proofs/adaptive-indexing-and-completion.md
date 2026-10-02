# Adaptive launch counts, random prefixes, and completion-set coverage

A common concern is that the number of launched tasks and the longest fully returned prefix are random and depend on past outcomes. The valid confidence constructions do not condition on those random indices.

For each type `i`, expose in advance an infinite iid tape `Z_i1,Z_i2,...` with law `p_i`. The scheduler reveals the next tape entry whenever it launches a type-`i` task. Let `N_it` be the number of such launches before decision `t`, and let `M_it <= N_it` be the longest chronological launch prefix whose audits have all returned.

Construct the finite-horizon simultaneous event used by the main theorem,

`G_i = { d(p_hat_i,n, p_i) <= r_i(n,delta) for every integer 1 <= n <= B T }`.

At most `B T` tasks are launched through horizon `T`, so every random index `N_it` or `M_it` lies in this deterministic range. The concentration proof is for all of those deterministic prefix lengths at once. Therefore, on `G_i`, the same inequality holds pathwise after substituting the random values. No claim that these indices are independent of the tape is required, and no conditional iid statement is used. An all-time event would require a different time-uniform radius and is not asserted here.

For prefix confidence, the first `M_it` entries are literally the first `M_it` entries of the tape, so the simultaneous event gives coverage directly.

For completion confidence, the actual empirical distribution of the first `N_it` tape entries is one feasible completion of the returned counts: assign every unresolved record its realized but not yet reported mark. Thus the actual `N_it`-sample empirical distribution belongs to the completion set. On `G_i`, the true law is within `r_i(N_it,delta)` of that feasible point and hence within the same radius of the completion set. This remains true when audit latency is an arbitrary function of the hidden mark, because no returned subset is treated as representative.

The intersection confidence set contains the truth whenever both component sets do. In the implementation both are certified by the same simultaneous tape event, so the intersection does not require an independence assumption between the two routes. A union bound is needed only across the explicitly indexed types/events used to construct the global event.
