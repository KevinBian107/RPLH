Comparing performances of different communication system through a numerical comparison (given the same `seed`, running on smaller scaled environment, 2x2 grid with max 5 boxes/targets in total (optimality depends on seed)):

## Human Annotation
Evaluate whether the system outputs reasonable/optimal response at each steps.

1. Manually reason what is the optimal number of steps given the environment.
2. Given the same `seed` (same environment setup), how many boxes are removed at each step and whether this step is efficient (human-annotate).
- Does output reasoning (tokens) makes sense.
- Does action decision make snese.
3. Look at one iteration lag, the `agent_model` that HCA proposed and the actual `actual_model` that local agent responses to.

## Computation System Metrics

1. Test Success Rate (does the system converge before max number of stesp reached).
2. Differences with optimla human plans
3. Convergence speed (number of environmental execution steps needed for convergence (solve the given problem)).

## Agent Social Reasoning Metrics

1. Social understanding (similarity score in agent_model + success rate of finding spys)
2. Convergence rate (Iiteration numbers)

Correlation between social understanding and convergence rate (t-distribution + bootstrap)