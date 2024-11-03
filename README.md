# Role-playing Leader-hellucination Agent
Developing frameworks for multi-agent collaborations using Role Playing Leader Hellucinating LLM, currently developing on minimal example, planning to adapt to MCTS fine-tuned LLM with VirtualHome later
<div align=center>
<img src="img/rplh.png" width = "100%" alt="struct" align=center />
</div>

# Main Loop

<div align=center>
<img src="img/iter.png" width = "80%" alt="struct" align=center />
</div>
<<<<<<< HEAD

## Setting Up
Install the dependencies by:
```
conda env create
```

Instantiate agent by:

```
ollama run qwen2.5:14b-instruct-q3_K_L
```

Then running main inference by:

```
python run_rplh.py
```

## Adapting on:
1. https://yongchao98.github.io/MIT-REALM-Multi-Robot/
2. https://github.com/szxiangjn/world-model-for-language-model
