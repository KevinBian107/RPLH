# Decentralized RPLH Instance
The decentralized RPLH instant, or `d_efficient`, is used for comparsion and validating with our Hallucinated version of RPLH system (`h_efficient` and `h_vanilla`). The decentralized system follows the general design of RPLH system with a few key implmentation details differences:

1. There is no central HCA agent nor is there a judge.

2. Attitude agent still exist.

3. To settle disagreement in all the loca agent, action is only executed when there are more than half of the agent that is operating in the environment (have box in it's grid and have action plan given by other agents) says "I Agree", else, the conversatoon continuses.

To compare to the Hallucination RPLH system, please refer to the following:

- [`h_efficient`](https://github.com/KevinBian107/RPLH/tree/master/rplh/h_efficient)

- [`h_vanilla`](https://github.com/KevinBian107/RPLH/tree/master/rplh/h_vanilla)
