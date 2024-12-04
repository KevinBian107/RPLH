# A Light-Weight Multi-Collaboration System
Using Language Models' **In-context Learning** abilities, we are developing frameworks for multi-agent collaborations system using Role-Playing Leader-Hallucinations system, or in short, RPLH. We know that as parameter increases, performance automatically increases. However, only limited works have being down on generalizing this sort of in-context learning ability into samll parameter models. In our work, we try to address this issue. We try to design an efficient `light-weight` system that is runable directly on a laptop computer 💻.

## Social Reasoning Agent's Perpective
Importantly, non of the agents actually see the full states of the environmnet (where all the boxes are at and where all the targets are at), so they must communicate about **what they can see** and **what they can do** to better collaborate with each other and do the actions. The purpose of designing an HCA agent is to make an hallucinating central agent that have a little bit mroe information compare to other agents and tells what all other what it sees and what the plan should be from that agent's perspective.

Here is an example of such partial information of describing what an specific local agent can see and do.

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

We hope to build agent that is caplable of **Social-reasoning** and expecting what the other agent should be doing. We are building a generalize world model for every single agent in this environment and hopefully moving a step closer to **Level-one agent**.

## RPLH Demos:
Here is a demo of our RPLH performing multi-agent resasoning with the first HCA (central) agent hallucinations about future steps:

- Conversation demos are [here](https://github.com/KevinBian107/RPLH/tree/master/demos/conversations)
- Actual running data are [here](https://github.com/KevinBian107/RPLH/tree/master/demos/converging_samples)

<div style="border: 1px solid #ccc; padding: 10px;">
    <iframe src="assets/rendering.html" width="800" height="600" style="border:none;"></iframe>
</div>

