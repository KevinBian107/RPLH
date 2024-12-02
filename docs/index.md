# A Light-Weight Multi-Collaboration System
Using Language Models' **In-context Learning** abilities, we are developing frameworks for multi-agent collaborations system using Role-Playing Leader-Hellucinating system, or in short, RPLH. We know that as parameter increases, performance automatically increases. However, only limited works have being down on generalizing this sort of in-context learning ability into samll parameter models. In our work, we try to address this issue. We try to design an efficient `light-weight` system that is runable directly on a laptop computer 💻.

## Agent's Perpective
Importantly, non of the agents actually see the full states of teh environmnet (where all the boxes are at and where all the targets are at), so they must communicate about **what they can see** and **what they can do** to better collaborate with each other and do the actions. The purpose of designing an HCA agent is to make an hellucinating central agent that have a little bit mroe information compare to other agents and tells what all other what it sees and what the plan should be from that agent's perspective.

Here is an example of such partial information

```bash
- Agent[0.5, 0.5]:
  - I am in square[0.5, 0.5],
  - I can observe ['box_green', 'target_orange'],
  - I can do one of the following action: ['move(box_green, square[1.5, 0.5])', 'move(box_green, square[0.5, 1.5])']
- Agent[0.5, 1.5]:
  - I am in square[0.5, 1.5],
  - I can observe ['target_red', 'box_green', 'target_green', 'target_green', 'target_purple', 'target_purple'],
  - I can do one of the following action: ['move(box_green, square[1.5, 1.5])', 'move(box_green, square[0.5, 0.5])', 'move(box_green, target_green)']
```

We hope to build agent that is caplable of ***Social-reasoning and expecting what the other agent should be doing***. We are building a generalize world model for every single agent in this environment and hopefully moving a step closer to ***Level-one agent***.

## RPLH Systems
**We hope to setup an systematic and modularzied way of doing multi-agent communication for less parametrized language models**. We have implemented multiple instances of the RPLH system dpending on `vanilla` or `efficient` version. Here is the main breakdown of the code:

```bash
rplh/
├── d_efficient/
│   ├── memory.py
│   ├── rplh_inference.py
├── h_efficient/
│   ├── env.py
│   ├── execution_checker.py
│   ├── memory.py
│   ├── rplh_inference.py
├── h_vanilla/
│   ├── env.py
│   ├── execution_checker.py
│   ├── memory.py
│   ├── rplh_inference.py
├── llm/
│   ├── language_model.py
│   ├── response_model.py
├── rendering/
│   ├── render_conversation.py
│   ├── render_states.py
│   ├── animations.py
└── inference.py
```
Note the following:

1. There are many shared features across all different type of communication system, which we ahve modularized into the `llm` folder for both using it in the vanilla and efficient models. In addition, because of implementation differences, files such as `env.py`, `execution_checker.py`, `memory.py`, and the main inference loop `rplh_inference.py` may be different, which is why each system has its own unique implementation.
2. The `d_efficient` (efficient implementation of decentralized framework) is a particular instance of the `h_efficient` system (our hallucination system) where the inference loops and memory module is adjusted.
3. Both `h_efficient` and `h_vanilla` are our hallucination system with slight differences in implementation, refer to these links for more details:
    - [`h_efficient`](https://github.com/KevinBian107/RPLH/tree/master/rplh/h_efficient)
    - [`h_vanilla`](https://github.com/KevinBian107/RPLH/tree/master/rplh/h_vanilla)
4. We have created inference functions in each of the system to be called seperately, or an direct interface can be called by specifying args on the root folder level of rplh (`inference.py`).


## RPLH Schematic:
The below is a graphical overview of our system:

![rplh](assets/rplh.png)

## RPLH Demos:
Here is a demo of our RPLH performing multi-agent resasoning with the first HCA (central) agent hellucinating about future steps:

- Conversation demos are [here](https://github.com/KevinBian107/RPLH/tree/master/demos/conversations)
- Actual running data are [here](https://github.com/KevinBian107/RPLH/tree/master/demos/converging_samples)

