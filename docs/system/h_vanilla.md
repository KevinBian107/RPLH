# Why Vanilla-RPLH System? 🤔
RPLH-original is the full representation of our construct where no limits has been posted on the structure of the output of the Language model, which comes with tradeoff of benifits and issues:

1. Have room for much more emergent behavior because teh context of the historical conversation is much richer.
    - More planning for future steps.
    - More reasoning about what other agents may be doing and thinking (attitude).
2. More computation resources needed as prompt is longer and history is longer.
3. Easier for LM to hallucinate and give unrealistic outputs (syntactic error).
    -  One iconic problem is where no dictionary bracket `{...}` is given.
    - There need to be more while loops for re-responding for the same condiion to get a better response, hence increasing teh compuatation loads.
4. Easier for LM to be effected by environmental checking prompt, resulting in some unexecutable iterations.
    - Slower convergence rate because of hallucination issue of the agent.

## Maintaining Syntactic Correctness
We have also performed a set of carefully designed engineering procedures to adjust the issues metioned above while maintaining the benifits of massive communication through conversations. LM is like an child where it needs very specific and detailed promppt for instructing it to  do things. We have designed and tested a set of prompt that is designed for the LM to maximize the freedom of communication while maintaining sytactic correctness.

1. Action checker
2. Json Checker
3. Markovian History or No History (prevent oerwhelming information and enviornmental disturbance)
    - At each round the agent is either completely fresh-agent or one-step Markovian agent.
4. Careful Instruction promts

Thus, though also can be used with smaller LLMs with really careful prompting and instruction, the vanilla RPLH version is more suitable for larger parametrized models such as GPT-4, which is the model that researcher usually use under these type of vanilla setting.