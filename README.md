# Role Playing Leader Hellucination System
Using Large Language Models' ***In-context Learning*** abilities, we are developing frameworks for multi-agent collaborations using **R**ole-**P**laying **L**eader-**H**ellucinating LLM, or in short, **RPLH-LLM**. We are currently developing on minimal example and we are planning to adapt to MCTS fine-tuned LLM with VirtualHome later.

## Simplified Schematic
<div align=center>
<img src="img/rplh.png" width = "70%" alt="struct" align=center />
</div>

We hope to build agent that is caplable of ***Social-reasoning and expecting what the other agent should be doing***. We are building a generalize world model for every single agent in this environment and hopefully moving a step closer to ***Level-one agent***.


## Communication Example
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

## Setting Up
Install the dependencies by:
```
conda env create
```

Instantiate LLM agent by (this need to be done in your system terminal directly, not in VScode):

```
ollama run qwen2.5:14b-instruct-q3_K_L
```

Then running main inference by:

```
python minimal_contained/run_rplh.py
```

## TODO:
1. Make synthetic check better and re better for getting json format answer (finished).
  - Try bigger models for base LLM.
  - Check complexity of the algorithm.
2. Adaptation to the VirtualHome environment
3. Use fine-tuned MCTS-World-Model LLM for task

## Adapting on:
1. https://yongchao98.github.io/MIT-REALM-Multi-Robot/
2. https://github.com/szxiangjn/world-model-for-language-model
