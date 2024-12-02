Our environment is adapted from [here](https://yongchao98.github.io/MIT-REALM-Multi-Robot/)

## Setting Up Inference Loop
First, we need to install the dependencies and create a separate conda environment:
```
conda env create
```

Download SLM from [here](https://ollama.com/library/qwen) and then instantiate small large language model (14B parameters qwen model) agent by:
```
ollama run qwen2.5:14b-instruct-q3_K_L
```
This need to be done in your system terminal directly, not in VScode.

## Running All Together
We can diretcly use the central running parser file to cerate the environment and tehn run the main inference loop by:
```
python rplh/inference.py --module_name "h_efficient" --model_name "qwen2.5:14b-instruct-q3_K_L"
```

## Create Environment Seperately
Create local MoveBox environment for running (depending on the version using) by:
```
python rplh/h_vanilla/env.py
```

Or setting up RPLH-efficient system by:
```
python rplh/h_efficient/env.py
```

## Inference Loops Seperately
We also descigned scripts to deirectly run each seperate system. We can run original vanilla RPLH inferene loop by:
```
python rplh/h_vanilla/rplh_inference.py -- model_name "qwen2.5:14b-instruct-q3_K_L"
```

or running the efficient RPLH inferene loop by:
```
python rplh/h_efficient/rplh_inference.py -- model_name "qwen2.5:14b-instruct-q3_K_L"
```

## Visualize Solutions
To visualize the reasoning process by rendering the conversation that each of the agent said:
```
python rplh/rendering/render_conversation.py
```
