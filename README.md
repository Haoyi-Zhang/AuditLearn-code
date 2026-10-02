# Optimal asynchronous speculation: delayed-audit learning artifact

This repository is the standalone proof and numerical artifact for
**Learning Speculative Schedules from Selectively Delayed Audits**.  The
repository basename is `optimal-asynchronous-speculation`; the name does not
claim a universally optimal scheduler.

The artifact studies a finite episodic scheduling model in which physical
verification can affect the current episode while the full task record used for
future learning arrives through a separate, possibly outcome-dependent audit
channel.  It contains the hand-proof supplement, exact finite checks, a frozen
diagnostic pilot, an all-dictionary stress campaign, and a nested horizon/backlog
sensitivity campaign.  It performs no model
inference and reports no decoding or hardware speedup.

## Established result

Under the assumptions listed below, an optimistic finite-model learner uses the
intersection of two valid confidence descriptions:

1. a confidence ball around the longest completely returned chronological
   prefix; and
2. a confidence neighborhood of a categorical completion polytope formed from
   all returned histograms and the total launch count.

The paper and `proofs/analysis.md` prove a launch-sensitive coupling and an
expected pseudo-regret bound whose delay contribution is

    min{chronological prefix debt, launch-normalized unresolved-mass exposure}.

For type `i`, launch-normalized unresolved-mass exposure is

    sum_t X_it u_it / max(1,N_it),

where `X_it` is the number of launches in episode `t`, `N_it` is the earlier
launch count, and `u_it` is the number of unreturned audits.  With at most one
same-type launch per episode and at most `U_i` outstanding audits, this term is
bounded by the sharp envelope `sum_{s=1}^{(N_i-1)_+} min{U_i,s}/s`, with
`Psi_U(0)=Psi_U(1)=Psi_0(n)=0`. Hence a
single old missing record can create linear chronological prefix debt while
incurring exactly harmonic unresolved-mass exposure.

The main result requires independent iid type tapes, a finite candidate dictionary
containing the truth, a finite causal policy class, bounded launches, bounded
centered cost, and a reliable full audit for every launched mark.  It does not
cover destroyed/censored marks, arbitrary misspecification, correlated tapes,
anonymous audits, a general efficient planning oracle, or a deployed decoder. A separate proved extension covers a declared typewise
TV approximation radius using inflated confidence sets and the conservative
coordinate sensitivity `B_i=L b_i`; the artifact has no off-grid empirical
campaign or inflated selector.

## Reproduction commands

Use a standard Python 3.10-or-newer interpreter on Linux.  Numerical code uses
only the Python standard library.  No package installation, network access,
external solver, child process, GPU library, model API, private dataset, or
hidden cache is required.  Each runner uses one worker and imposes a 3584 MiB
address-space cap.

From this directory, the complete bounded workflow is available as:

```sh
./reproduce.sh repro-results
```

The script expands to the following commands:

```sh
OUT=repro-results
python run.py check --output "$OUT"
python run.py pilot --output "$OUT"
for i in 0 1 2 3 4 5 6 7; do
  python run.py stress --output "$OUT" --shard-index "$i" --shard-count 8
done
python run.py merge-stress --output "$OUT" --shard-count 8
for i in 0 1 2 3; do
  python run.py scale --output "$OUT" --shard-index "$i" --shard-count 4
done
python run.py merge-scale --output "$OUT" --shard-count 4
python run.py verify --output "$OUT"
python report.py --results "$OUT"
```

The commands have the following roles.

- `check` regenerates exact policy values and all finite checks.
- `pilot` regenerates the complete 252-run pilot, including every episode.
- the eight `stress` shards partition the 108 truth--seed cells deterministically;
  `merge-stress` rejects missing, duplicated, or malformed cells before writing
  the 1,512-run merged campaign;
- the four `scale` shards partition the 12 truth--seed cells deterministically;
  `merge-scale` performs the corresponding completeness checks before writing
  the 600-run merged campaign;
- `verify` reconciles every pilot run summary with its episode trace; and
- `report.py` reconciles curve prefixes and generates every paper-side table and
  plot-data fragment from the retained CSV files.

An interrupted campaign can resume at its missing shard without changing any
random stream. Direct unsharded `stress` and `scale` commands remain supported,
but the sequence above is the documented clean-reproduction path because every
long step stays bounded.

A smaller optional execution test is available:

```sh
python run.py smoke --output smoke-results
```

It is not part of the reported campaigns.  Do not run the exact checks with
Python's `-O` flag because assertions are part of the finite checker.

All `run.py` commands accept an alternate output directory. Sharded campaign
files include their zero-padded shard index and count; merge commands validate
all expected files and write the canonical merged CSVs. Do not mix shards from
different configurations or shard counts.

The artifact does not depend on the paper directory.  The paper copies only the
small generated TeX/data fragments after byte comparison.

## Files

- `reproduce.sh` — documented deterministic shard-and-merge workflow.
- `run.py` — bounded single-worker command-line runner.
- `report.py` — deterministic aggregation and paper-fragment generator.
- `src/model.py` — finite fork semantics and an independently structured
  backward oracle.
- `src/learning.py` — immutable audit views, confidence widths, completion
  geometry, and policy selection.
- `tests/check_exact.py` — rational enumeration, geometry/counting checks,
  audit-order checks, and information-isolation checks.
- `proofs/analysis.md` — complete hand-proof supplement.
- `inputs/pilot.json` — frozen diagnostic pilot configuration.
- `inputs/stress.json` — frozen all-dictionary stress configuration.
- `inputs/scaling.json` — frozen nested horizon/backlog sensitivity configuration.
- `results/` — complete retained evidence and generated paper fragments.
- `claim_evidence_ledger.csv` — claim-to-proof/check/result mapping.
- `external_resources.csv` and `source_passages.csv` — scholarly provenance and
  the boundaries used in the literature comparison.
- `reference_audit.csv` — all 85 manuscript references, topical clusters,
  citation status, and verification scope.
- `LICENSE` — license for this artifact's own code and text.

## Finite scheduling model

Root type 0 has marks `(Y,D)` ordered as
`(0,1),(1,1),(0,2),(1,2)` with law

    ((1+rho)/4, (1-rho)/4, (1-rho)/4, (1+rho)/4).

Child types 1 and 2 have `(success,duration)` ordered as
`(0,1),(0,2),(1,1),(1,2)` with law

    ((1-p)/2, (1-p)/2, p/2, p/2).

The root starts at physical time zero.  At time one, root arrivals are processed
before the policy waits or launches one child.  A root arrival starts a
two-tick fallback.  A correct successful child is useful only if it finishes
strictly before fallback; ties count as waste.  Cost is completion time plus
one quarter per wasted optional launch.  Centering by mandatory root duration
leaves a residual cost in `[0,9/4]`.

There are three boundary observations—early branch 0, early branch 1, and root
pending—and three actions—wait, launch child 0, launch child 1.  The 27
deterministic policies are ordered lexicographically.  The model dictionary is

    rho in {-3/4,0,3/4},
    p_0,p_1 in {1/4,1/2,3/4},

for 27 laws.  All 64 latent mark triples are enumerated for every model-policy
pair.

The learner receives immutable `AuditView` objects containing launch counts,
returned histograms, and chronological-prefix histograms.  It does not receive
the truth index, a pending payload, an unlaunched child mark, future worlds, or
the simulator's report queue.  Known-law and product-marginal reference
policies and truth-exclusion diagnostics remain outside the selector.

## Audit schedules

The pilot uses outcome-dependent reporting.  A launched record from episode
`t` is available before boundary `t+lag+1` when its branch/success signal equals
one and before boundary `t+1` otherwise.  The pilot covers lags 0, 16, and 64.

The stress campaign uses two schedules:

- `outcome-lag-64`: the same mark-dependent schedule at lag 64;
- `single-blocker`: the first launched mark of each type is withheld until
  after the learning horizon and every later mark returns at the next boundary.

After the last learning episode, queued audits are drained solely to verify
full-audit closure.  Drained records never affect a policy decision.

## Exact checks

`results/exact_checks.json` currently records:

- 27 oracle-minimum matches;
- 729 exact policy values and 729 independently structured
  value/waste/launch triplets;
- 19,683 launch-coupling inequalities;
- 1,973 completion-polytope projection cases;
- 3,320 four-category states comparing the exact distance formula with the
  selector implementation;
- 209 exact fractional/integer completion-diameter cases;
- 26,244 chronological-prefix exposure cases;
- 26,244 completion-exposure cases;
- 8,721 saturated outstanding-envelope cases, including `n=0`, `n=1`,
  `U=0`, and `U>=n` boundary regimes;
- 9,216 unlaunched-mark noninterference cases;
- 96 audit-order states; and
- five selector information-boundary checks; and
- one exact known-radius binary regression for center `0.50`, `r=0.01`,
  `epsilon=0.10`, cover point `0.60`, and selected candidate `0.39`.

The checks also retain three exact counterexamples:

- the product-marginal policy has regret `63/128` per episode under the specified
  positive-correlation law;
- a selected returned subset can have success fraction zero while the full
  length-80 tape has empirical fraction one half; and
- one unreturned audit can coexist with prefix debt 256, invalidating a bound
  that replaces chronological debt by missing-record count.

Finite checks complement the general proofs; they do not establish them for all
alphabets, trees, or reporting processes.

## Diagnostic pilot

The frozen pilot comprises 252 runs, 129,024 episodes, and 4,032 retained curve
points: three diagnostic truths, three reporting lags, four seeds, seven
methods, and horizon 512.

Every run summary reconciles with the complete episode trace.  The deliberately
invalid completed-only control excludes the true model in 213 dependent episode
records.  Prefix and intersection confidence have zero exclusions in the
pilot.  At truth 20 and lag 64, mean cumulative pseudo-regret is 51.26 for
prefix confidence, 27.24 for intersection confidence, and 5.92 for the plug-in
baseline.  The baseline's favorable result is retained; there is no universal
dominance claim.

`results/bound_values.csv` shows that the displayed worst-case theorem bound is
numerically vacuous relative to the trivial bound at this horizon.  This is a
negative calibration, not a failed reproduction.

## All-dictionary stress campaign

The stress campaign comprises 1,512 runs and 774,144 episodes: all 27 truths,
four seeds, seven methods, two audit schedules, and horizon 512.  It retains
24,192 curve rows.

For `single-blocker`, the maximum unreturned count is one while maximum prefix
debt is 511.  Across 108 paired truth-seed cells, intersection confidence has
lower pseudo-regret than prefix confidence in 76 cells, ties in 27, and is
worse in five.  Its mean paired difference is `-65.0811`.  Mean cumulative
pseudo-regret is 72.59 for prefix confidence and 7.51 for intersection
confidence.  The deliberately invalid `completed` returned-sample control has mean 7.17
under this non-mark-selected schedule, and the plug-in baseline has mean 9.35;
that favorable control result does not make its confidence construction valid
under outcome-dependent returns.

For `outcome-lag-64`, intersection is better in 69 of 108 cells, ties in 31,
and is worse in eight, with mean paired difference `-4.7967`.  Mean regret is
15.03 for prefix and 10.24 for intersection.  Completed-only confidence records
2,234 truth exclusions and 2,200 safe fallbacks.

These are descriptive paired outcomes for the explicit finite dictionary, not
confidence intervals for a deployment population.

## Horizon/backlog sensitivity campaign

The prespecified scaling campaign adds 600 runs and 476,160 episodes over
horizons 128, 256, 512, 1024, and 2048; blocker counts one and four; three
truths; four seeds; and five methods. Shorter runs are exact prefixes of a
common 2,048-episode world stream. Every row attains the sharp descriptor
envelope `sum_i Psi_U(N_i)` to numerical tolerance. The regression explicitly
retains 80 rows with at least one unlaunched task type and verifies that each
such type contributes `Psi_U(0)=0`.

Across all ten blocker/horizon slices, intersection is never worse than prefix
in any of the 12 paired truth-seed cells. For one blocker, mean prefix regret
changes from 27.08 at horizon 128 to 433.33 at 2048, while intersection changes
from 6.47 to 10.67. These are finite-dictionary sensitivity results, not an
empirical asymptotic-rate estimate.

## Result files

`episode_trace.csv` contains every pilot episode.  Its `regret` column is the
exact-law policy-value gap, not the realized sample-cost difference.
`expected_waste` is a policy expectation and `realized_cost` is the sampled
physical objective.  Counts `n0..n2`, `m0..m2`, and `u0..u2` are measured before
the current episode's launches.

`runs.csv` and `curves.csv` aggregate the pilot.  `stress_runs.csv` and
`stress_curves.csv` retain the stress campaign; `scale_runs.csv` retains the
sensitivity campaign.  `aggregate.csv`,
`stress_aggregate.csv`, `stress_dictionary_summary.csv`, and
`stress_pairwise.csv` are deterministic descriptive summaries.  TeX and `.dat`
files in `results/` are generated directly from those summaries.

Execution JSON files record process CPU, wall time, and peak resident memory.
For the retained reference logs, the eight stress-shard CPU fields sum to
`29.595921274` seconds and the four scaling-shard fields sum to `21.04284331`
seconds. Their largest shard wall times are `3.86450011100001` and
`5.31110579200003` seconds. Across those shards and their merge commands, the
largest host-reported command peak is `124584` KiB. The records do not identify
the processor model. These values are environment-specific diagnostics; CPU
sums are sequential accounting, wall times are not deterministic, and RSS peaks
are not additive or a whole-session trace.

## Interpretation and limitations

The experiments use common random worlds to reduce paired noise.  Unlaunched
marks are sampled only for world scoring and remain hidden from learners.
Reported sample standard deviations summarize four seeds; they are not
independent workload confidence intervals.

Raw waste is not cost regret.  A known-law policy can intentionally launch
work that is often unused because its latency benefit outweighs the waste
price.  Similarly, a narrower confidence set at a fixed history need not win on
every sample path because it changes later launches and observations.

The learning architecture has close antecedents in delayed online learning,
reward-dependent-delay confidence methods, combinatorial/triggered bandits,
and online speculative selection.  The claimed result is narrower: exact
categorical completion geometry and a common-term exposure decomposition that
produces the minimum of chronological prefix debt and launch-normalized
unresolved-mass exposure under reliable typed full audits. Recent capacity-constrained delayed
optimization already uses a harmonic backlog cost; harmonic weighting itself is
not claimed as new.  No component technique is
claimed to be universally new.

The artifact has been self-audited but not independently reviewed.  A
successful command proves local execution and internal consistency, not the
scientific correctness of every argument.  Before external use, human authors
must recheck venue, authorship, disclosure, originality, and repository rules.

## Release reproduction path

The complete release workflow is:

```bash
python reproduce_release.py repro-results
```

It executes exact checks and the pilot, then the documented eight-shard stress
chain and four-shard scaling chain with their merge completeness checks,
followed by pilot verification, deterministic report generation, smoke, and
scope-matched validators. The artifact exposes no `offgrid` command and has no
off-grid input or result.

For a quick installation and serialization check without regenerating the two
long campaigns:

```bash
python reproduce_release.py quick-results --skip-long
```

The quick report explicitly labels its scope and cannot be used as evidence
that stress or scaling were regenerated. It no longer attempts to resolve a
nonexistent off-grid command.

When this repository is extracted without the sibling `paper/` directory, the
release validator explicitly records that the bibliography audit was not
performed; it does not report that absent paper-side check as a pass. The same
validator performs the full citation-key audit inside the complete project.

The serialized method label `completed` is retained to preserve the identity of
the frozen tables. It denotes the deliberately invalid control that treats
returned records as an iid sample; it is not a completion-polytope-only learner.
The valid learners reported here are `prefix` and `intersection`. The
known-radius dictionary extension is theory plus an exact arithmetic regression,
not an empirical campaign.
