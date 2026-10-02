# Assumption and limitation matrix

| Assumption | Where used | What fails without it | Covered extension / evidence | Not claimed |
|---|---|---|---|---|
| Typed potential-mark tapes are iid across launch index | simultaneous finite-horizon concentration and oracle comparison | arbitrary drift/correlation invalidates the confidence radii | exact adaptive-indexing argument for `1 <= n <= B T` | nonstationary or adversarial marks |
| Audits preserve identity and eventually reveal the complete categorical mark | completion set and chronological prefix | anonymous, corrupted, or permanently missing records need another observation model | outcome-dependent and front-blocker latency | censoring/deletion robustness |
| True law is in a finite dictionary | nominal optimism and finite planning | nominal confidence can exclude all dictionary points | proved known-radius TV-cover extension with `B_i=L b_i`; exact binary arithmetic regression | off-grid empirical validation or adaptation to unknown misspecification |
| Policy class and planning oracle are finite/available | optimistic policy selection | generic scheduling may be computationally intractable | explicit planning-error term; exact enumeration in the artifact | polynomial-time oracle for general systems |
| Per-episode loss and launches are bounded | coupling and failure-event conversion | sensitivity and regret can be unbounded | constants are explicit in the theorem | heavy-tailed unbounded service costs |
| Full visible histories of two environments are equal in law during an audit blackout | conditional necessity statement only | physical observations or scalar costs may distinguish the environments | conditional two-environment averaging argument | an unconditional blackout lower bound for the finite fork |
| Simulator benchmark is synthetic and finite | empirical mechanism checks | no direct evidence about decoder/hardware throughput | exhaustive finite checks and pilot/stress/scaling campaigns | deployed speedup, energy, or latency claims |
| Seeds are Monte Carlo replicates, not sampled deployments | uncertainty summaries | population inference would be unsupported | paired finite-cell summaries and seed-level variation | real-world confidence intervals |

The paper's positive statements are conditional on the left-hand assumptions.
The artifact checks implementation consistency under those assumptions; it does
not convert them into empirical facts about an external system.
