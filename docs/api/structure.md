## RPLH Systems
We hope to setup an **systematic** and **modularzied** way of doing multi-agent communication for **less parametrized** language models. We have implemented multiple instances of the RPLH system dpending on `vanilla` or `efficient` version. Here is the main breakdown of the code:

```bash
rplh/
├── env/
│   └── env.py
├── d_efficient/
│   ├── memory.py
│   └── rplh_inference.py
├── h_efficient/
│   ├── execution_checker.py
│   ├── memory/
│   │   ├── memory_standard.py
│   │   └── memory_agent.py
│   ├── rplh_inference.py
│   └── rplh_agent_inference.py
├── h_vanilla/
│   ├── execution_checker.py
│   ├── memory.py
│   └── rplh_inference.py
├── llm/
│   ├── language_model.py
│   └── response_model.py
├── rendering/
│   ├── render_conversation.py
│   ├── render_states.py
│   └── animations.py
├── evaluation/
│   └── evaluation.py
├── test.py
└── inference.py

```
Note the following:

1. There are many shared features across all different type of communication system, which we ahve modularized into the `llm` folder, `env` folder, or the `rendeiring` folder for both using it in the vanilla and efficient models. In addition, because of implementation differences, files such as `execution_checker.py`, `memory.py`, and the main inference loop `rplh_inference.py` may be different, which is why each system has its own unique implementation.
2. The `d_efficient` (efficient implementation of decentralized framework) is a particular instance of the `h_efficient` system (our hallucination system) where the inference loops and memory module is adjusted.
3. Both `h_efficient` and `h_vanilla` are our hallucination system with slight differences in implementation, refer to these links for more details:
    - [`h_efficient`](https://github.com/KevinBian107/RPLH/tree/master/rplh/h_efficient)
    - [`h_vanilla`](https://github.com/KevinBian107/RPLH/tree/master/rplh/h_vanilla)
4. We have created inference functions in each of the system to be called seperately, or an direct interface can be called by specifying args on the root folder level of rplh (`inference.py`).


## Graphic Schematic:
The below is a graphical overview of our system:

![rplh](../assets/rplh.png)