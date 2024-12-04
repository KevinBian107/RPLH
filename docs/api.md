## Setting Up
First, we need to install the dependencies and create a separate conda environment:

```python
conda env create
```

Our environment is adapted from [here](https://yongchao98.github.io/MIT-REALM-Multi-Robot/)

Download SLM from [here](https://ollama.com/library/qwen) and then instantiate small large language model (14B parameters qwen model) agent by:

```python
ollama run qwen2.5:14b-instruct-q3_K_L
```

This need to be done in your system terminal directly, not in VScode.

## Inference Pipeline
We can diretcly use the central running parser file to cerate the environment and tehn run the main inference loop by:

```python
python rplh/inference.py --module_name "h_efficient" --model_name "qwen2.5:14b-instruct-q3_K_L"
```

For communication system, we can choose `module_name` between `h_efficient`, `d_efficient`, or `h_vanilla`. Notice that for `h_vanilla`, we are connecting to GPT-4o-mini's backend, but switching to ollama like other systems is doable as well. For `model_name`, for demonstartion we use `qwen2.5:14b-instruct-q3_K_L` or `gpt-4o-mini`.

## Testing Systems
Conduct evaluation using our system by:

```python
python rplh/test.py --module_name "h_efficient" --model_name "qwen2.5:14b-instruct-q3_K_L" --num_trials 5 --box_num_upper_bound 1 --seed 0
```

## Visualize Solutions
To visualize the reasoning process by rendering the conversation that each of the agent said:

```python
python rplh/rendering/render_conversation.py
```

## Customize By Runing Seperately
### Create Individual Environment
Create local MoveBox environment for running (depending on the version using) by:

```python
python rplh/h_vanilla/env.py
```

Or setting up RPLH-efficient system by:

```python
python rplh/h_efficient/env.py
```

### Individual Inference Loops
We also descigned scripts to deirectly run each seperate system. We can run original vanilla RPLH inferene loop by:

```python
python rplh/h_vanilla/rplh_inference.py --model_name "qwen2.5:14b-instruct-q3_K_L"
```

or running the efficient RPLH inferene loop by:

```python
python rplh/h_efficient/rplh_inference.py --model_name "qwen2.5:14b-instruct-q3_K_L"
```