Comparing performances of different communication system through a numerical comparison (given the same `seed`, running on smaller scaled environment, 2x2 grid with max 5 boxes/targets in total (optimality depends on seed)):

## LM Setups
We used different setup combination between systems and models, which we describe them below:

| **LM/System**        | **H_Efficient** | **Dec_Efficient** | **H_Vanilla** |
|-----------------------|-----------------|--------------------|---------------|
| Ollama-qwen           | ✅              | ✅                 |               |
| GPT-4o-mini           |                 | ✅                 | ✅            |

Similaerly, the Agent model reasoning module has specific setting as well:

| **Reasoning/System** | **H_Efficient** | **Dec_Efficient** | **H_Vanilla** |
|-----------------------|-----------------|--------------------|---------------|
| Standard Reasoning    | ✅              |                    |              |
| Agent Reasoning       | ✅              |                    |              |


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
4. Critically, the number of agreement to see if agent based reasoning actually help.