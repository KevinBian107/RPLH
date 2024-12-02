## Numerical Evaluation
Comparing performances of different communication system through a numerical comparison:

1. Test Success Rate (does the system converge before max number of stesp reached).
2. Convergence speed (number of execution steps needed for convergence (solve the given problem)).
3. Number of API calls.

## Naturalistic Reasoning Evaluation
Evaluate whether the system outputs reasonable/optimal response at each steps.

1. Given the same `seed` (same environment setup), how many boxes are removed at each step and whether this step is efficient (human-annotate).
- Does output reasoning (tokens) makes sense.
- Does action decision make snese.