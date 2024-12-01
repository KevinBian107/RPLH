# Role Playing Leader Hellucination (RPLH) Light-Weight Multi-Collaboration System
Using Large Language Models' ***In-context Learning*** abilities, we are developing frameworks for multi-agent collaborations/commetning system using **R**ole-**P**laying **L**eader-**H**ellucinating LLM, or in short, **RPLH**. We know that as parameter increases, performance automatically increases. However, only limited works have being down on generalizing this sort of **in-context** learning ability into samll parameter models. In our work, we try to address this issue. We try to design an efficient `light-weight` system that is runable directly on a laptop computer 💻.

<div align=center>
<img src="demos/img/rplh.png" width = "50%" alt="struct" align=center/>
</div>

Importantly, non of the agents actually see the full states of teh environmnet (where all the boxes are at and where all the targets are at), so they must communicate about **what they can see** and **what they can do** to better collaborate with each other and do the actions.

We hope to build agent that is caplable of ***Social-reasoning and expecting what the other agent should be doing***. We are building a generalize world model for every single agent in this environment and hopefully moving a step closer to ***Level-one agent***.

## RPLH Demos:
Here is a demo of our RPLH performing multi-agent resasoning with the first HCA (central) agent hellucinating about future steps:

<div style="width: 100%; padding: 5px; display: flex; justify-content: center;">
  <video controls autoplay style="width: 70%; height: auto;" muted>
    <source src="demos/img/rplh_demo1.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>


## Setting Up Inference Script:

First, we need to install the dependencies and create a separate conda environment:
```
conda env create
```

Download SLM from: https://ollama.com/library/qwen and then instantiate small large language model (14B Ollama model qwen2.5:14b-instruct-q3_K_L) agent by (this need to be done in your system terminal directly, not in VScode):
```
ollama run qwen2.5:14b-instruct-q3_K_L
```

Create local MoveBox environment for running (depending on the version using) by:
```
python rplh_vanilla/env.py
```

Or setting up RPLH-efficient system by:
```
python rplh_efficient/env.py
```

Then running original vanilla RPLH inferene loop by:
```
python rplh_vanilla/rplh_inference.py -- model_name "qwen2.5:14b-instruct-q3_K_L"
```

or running the efficient RPLH inferene loop by:
```
python rplh_efficient/rplh_inference.py -- model_name "qwen2.5:14b-instruct-q3_K_L"
```

To visualize the reasoning process by rendering the conversation that each of the agent said:
```
python rendering/render_conversation.py
```

## Environment Aaapted From:
1. https://yongchao98.github.io/MIT-REALM-Multi-Robot/
