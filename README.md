# Role Playing Leader Hellucination System
Using Large Language Models' ***In-context Learning*** abilities, we are developing frameworks for multi-agent collaborations using **R**ole-**P**laying **L**eader-**H**ellucinating LLM, or in short, **RPLH-LLM**. We are currently developing on minimal example and we are planning to adapt to MCTS fine-tuned LLM with VirtualHome later.
- Judge also serve as an syntactic checker as well.

## Demo:
Here is a demo of our RPLH performing multi-agent resasoning with the first HCA (central) agent hellucinating about future steps:

<div style="width: 100%; padding: 5px; display: flex; justify-content: center;">
  <video controls autoplay style="width: 60%; height: auto;" muted>
    <source src="demos/img/rplh_demo1.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>


## TODOs:
1. Make synthetic check better and re better for getting json format answer (jusge get stuck).
  - Try bigger models for base LLM.
  - How to handle edge cases when the action that one of the action is not doable? If local agent action not doable, fall back to HCA action.
  - `Function calling` LLM?
2. "Finetune" RPLH details. 
3. Add logging of each step of the process for visualization.
4. Modularize code with RPLH class and good documentation baked into the strcuture and organization of the code.
5. Evaluation metrics and run numbers of experiment runs.
  - Run scalable experiments with 2x2, 2x4, 4x4, 4x8 just as in env create.
6. Add different color for different agent visualized.

## Simplified Schematic
<div align=center>
<img src="demos/img/rplh.png" width = "60%" alt="struct" align=center/>
</div>

We hope to build agent that is caplable of ***Social-reasoning and expecting what the other agent should be doing***. We are building a generalize world model for every single agent in this environment and hopefully moving a step closer to ***Level-one agent***.


## Communication
Example demonstrating the communication. This is an fun example of lcoal agent complaining of the action given by others

```
------###------###------HCA_0------###------###------: 
...
Considering these points:
{
    "Agent[0.5, 0.5]" : "",
    "Agent[0.5, 1.5]" : "move(box_blue, square[0.5, 0.5])",
    "Agent[1.5, 0.5]" : "move(box_red, target_red)",
    "Agent[1.5, 1.5]" : ""
}

------###------###------LOCAL_0_ROW_0_COL_0------###------###------: 
My objections are:
- The given action plan misses out on actions for Agent[0.5, 0.5], which includes myself.
- It's unclear how moving box_blue from square [0.5, 1.5] aligns with the observable list provided for that square in the context of coordinating all agents towards their respective targets.
...
{"Agent[0.5, 0.5]" : "move(box_orange, target_orange)"}

------###------###------JUDGE_0_ROW_0_COL_0------###------###------: 
 {"Agent[0.5, 0.5]":"move(box_orange, target_orange)"} 
 
------###------###------LOCAL_0_ROW_0_COL_1------###------###------: 
 I Agree

 ------###------###------JUDGE_0_ROW_0_COL_1------###------###------: 
 {"Agent[0.5, 0.5]":"move(box_orange, target_orange)"} 
 
------###------###------LOCAL_0_ROW_1_COL_0------###------###------: 
Not agreed. The current action plan only includes Agent[0.5, 0.5] moving the box_orange to its target, which was already executed in the previous step.
...
Proposed updated action plan:
{"Agent[1.5, 0.5]":"move(box_red, square[0.5, 0.5])"} 
```

## Hellucination
Here is an example of how the HCA agent hellucinate:
```
------###------###------HCA_0------###------###------: 
 - Attitude of Agent[0.5, 0.5]: Proactive and detail-oriented. Given the actions available in its current state, it is focused on moving boxes that can be moved directly out of the way to clear space for other tasks or towards their respective targets if possible.

    - Reaction: The agent will likely respond positively to any instructions that align with clearing its immediate area, especially those involving moving a box_blue, as this aligns with its proactive nature.
    
    - Commanding action: As the central coordinator, I would instruct Agent[0.5, 0.5] to prioritize moving boxes out of its way towards their targets whenever possible.

- Attitude of Agent[1.5, 0.5]: Resourceful and adaptable. This agent has a variety of actions available at different squares and can move various types of boxes.

    - Reaction: The agent will likely appreciate instructions that involve multiple types of boxes being moved simultaneously but may become cautious if instructed to perform tasks that it perceives as not leading directly towards the goal state.
    
    - Commanding action: I would instruct Agent[1.5, 0.5] to move box_blue from its current location to a square where it can be closer to its target.

- Attitude of Agent[1.5, 1.5]: Focused on tasks that require direct attention. With only one type of box within reach, the agent is likely to act directly and with precision when given instructions.
    
    - Reaction: The agent will positively respond to instructions involving moving the available box_purple towards its target or clearing space for other actions if instructed to do so.
    
    - Commanding action: I would instruct Agent[1.5, 1.5] to move the box_purple closer to a square where it can eventually be moved directly to its target.

- Attitude of Agent[0.5, 1.5]: Limited by the current state but willing to assist once additional actions become available.

    - Reaction: The agent will wait for instructions that involve moving towards an achievable goal or clearing space that could lead to further actions being available.
    
    - Commanding action: I would instruct Agent[0.5, 1.5] to remain in its current state until more actions become possible.

- Hallucination of future 5 steps...:

Given the current states and the proactive attitudes towards achieving the goal:
1. In step one, moving box_blue from square [0.5, 0.5] to square [0.5, 1.5].
2. Following this action, Agent[0.5, 1.5], now having a valid action, will move box_blue towards its target.
3. Simultaneously, Agent[1.5, 0.5] moves another box_purple closer to square [0.5, 0.5].
4. Next, moving the newly positioned box_green from step one or two directly towards a square aligned with its goal state.
5. Finally, Agent[1.5, 1.5], having moved box_purple in an earlier step, now moves it to its target.

- Plan for immediate next step:

{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_purple, square[0.5, 0.5])"} 
 ```

## Setting Up
Install the dependencies by:
```
conda env create
```

Download SLM from: https://ollama.com/library/qwen

Instantiate SLM agent by (this need to be done in your system terminal directly, not in VScode):
```
ollama run qwen2.5:14b-instruct-q3_K_L
```

Create local environment first by:
```
python env_create.py
```

Then running main inference by:
```
python rplh/run_rplh.py
```

To visualize the reasoning process:
```
python rplh/vis_conversation.py
```

## Adapting on:
1. https://yongchao98.github.io/MIT-REALM-Multi-Robot/
2. https://github.com/szxiangjn/world-model-for-language-model
