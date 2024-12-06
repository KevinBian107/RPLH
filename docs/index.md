# A Light-Weight Agetnt-Based Multi-Collaboration System
We hope to build agent that is caplable of **social-reasoning** and expecting what the other agent should be doing. We are building a generalize agent model for every single agent in this environment and hopefully moving a step closer to **level-one agent**.

Using Language Models' **In-context Learning** abilities, we are developing frameworks for multi-agent collaborations system using Role-Playing Leader-Hallucinations system, or in short, RPLH. We know that as parameter increases, performance automatically increases. However, only limited works have being down on generalizing this sort of in-context learning ability into samll parameter models. In our work, we try to address this issue. We try to design an efficient `light-weight` system that is runable directly on a laptop Macbook 💻.

## Intelligence In Information & Data Flow
We believe that the key of collaboration, or of solving issues in such partial observable and actuabtable environment layes in communication and the data/inofrmation we can extract from such communication to gradually build up a world model.

## Social Reasoning Agent
In decentralzie environment, usually each agent must communicate about **what they can see** and **what they can do** to better collaborate with each other and do the actions because of the partial observability situation. Here is an example of such partial information of describing what an specific local agent can see and do.

```
Agent[0.5, 0.5]:
- I am in square[0.5, 0.5],
- I can observe ['box_green', 'target_orange'],
- I can do one of the following action: ['move(box_green, square[1.5, 0.5])', 'move(box_green, square[0.5, 1.5])']
```

Differently, we aer using a **fully-observable environment**, meaning that all agent can see everything (where all the boxes are at and where all the targets are at). Howveer, the environmnet still remains partially actuatable as there are limitations to what each agent can do. However, even with this setting, a common type of problem in usual decentralzied collaboration setting (each agent has their own perspective and doesn't just do whatever the central agent tells them to do) is that it is extremely hard for different agents to come to an agreement on the plan, especially when the number of agent scales.

Our system propose a novel approach to solve this problem where each HCA agent try to conduct social reasoning of what the other agent would think, do, and react to it's plan and recieve actual sensory feedbacks from local agents to understand what is the problem with teh previous reasoning. Idealy, through longer iyerations, the HCA would gradually build an more correct model of other agents. This situation improves complexity even more when we have different local agent playing this role of the HCA agent, meaning that they can incorporate their attitudes into giving actions.

## RPLH Demos:
Here is a demo of our RPLH performing multi-agent resasoning with the first HCA (central) agent hallucinations about future steps:

- Conversation demos are [here](https://github.com/KevinBian107/RPLH/tree/master/demos/conversations)
- Actual running data are [here](https://github.com/KevinBian107/RPLH/tree/master/demos/converging_samples)

<div style="border: 1px solid #ccc; padding: 10px;">
    <iframe src="assets/rendering.html" width="800" height="600" style="border:none;"></iframe>
</div>

