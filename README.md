# Role Playing Leader Hellucination (RPLH) Multi-Commenting System

Using Large Language Models' ***In-context Learning*** abilities, we are developing frameworks for multi-agent collaborations using **R**ole-**P**laying **L**eader-**H**ellucinating LLM, or in short, **RPLH-LLM**. We are currently developing on minimal example and we are planning to adapt to MCTS fine-tuned LLM with VirtualHome later.
- Judge also serve as an syntactic checker as well.
- As parameter increases, performance increases

<div align=center>
<img src="demos/img/rplh.png" width = "50%" alt="struct" align=center/>
</div>

We hope to build agent that is caplable of ***Social-reasoning and expecting what the other agent should be doing***. We are building a generalize world model for every single agent in this environment and hopefully moving a step closer to ***Level-one agent***.

## Demo:
Here is a demo of our RPLH performing multi-agent resasoning with the first HCA (central) agent hellucinating about future steps:

<div style="width: 100%; padding: 5px; display: flex; justify-content: center;">
  <video controls autoplay style="width: 70%; height: auto;" muted>
    <source src="demos/img/rplh_demo1.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>


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

Then running original full RPLH main inference by:
```
python rplh_full/run_rplh.py
```

or running the efficient version

```
python rplh_efficient/run_rplh.py
```

To visualize the reasoning process:
```
python rplh/vis_conversation.py
```

## Adapting on:
1. https://yongchao98.github.io/MIT-REALM-Multi-Robot/
