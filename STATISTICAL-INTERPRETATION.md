# Statistical interpretation

The truth, policy, horizon, and delay grids are deliberately enumerated finite benchmarks. They are not random samples from a deployment population. Random seeds represent Monte Carlo replication of the specified simulator.

Accordingly:

1. comparisons use identical truth/seed/delay cells whenever methods are paired;
2. tables report counts, means, medians, quantiles, worst cells, and seed-level Monte Carlo standard errors where meaningful;
3. no benchmark-cell `p`-value is interpreted as evidence about an unspecified real-world population;
4. claims are restricted to the configured finite benchmark and theorem assumptions;
5. unfavorable cells and methods that outperform the proposed learner are retained.

A confidence-set coverage statement in this artifact concerns the mathematical simulator and the stated event. It is not a calibration claim for deployed language-model or hardware data.
