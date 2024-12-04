## RPLH Systems
We hope to setup an **systematic** and **modularzied** way of doing multi-agent communication for **less parametrized** language models. We have implemented multiple instances of the RPLH system dpending on `vanilla` or `efficient` version. Here is the main breakdown of the code:

```bash
rplh/
├── d_efficient/
│   ├── memory.py
│   ├── rplh_inference.py
├── h_efficient/
│   ├── env.py
│   ├── execution_checker.py
│   ├── memory/
│     ├── memory.py
│     ├── memory_partial.py
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
├── test.py
└── inference.py
```
Note the following:

1. There are many shared features across all different type of communication system, which we ahve modularized into the `llm` folder for both using it in the vanilla and efficient models. In addition, because of implementation differences, files such as `env.py`, `execution_checker.py`, `memory.py`, and the main inference loop `rplh_inference.py` may be different, which is why each system has its own unique implementation.
2. The `d_efficient` (efficient implementation of decentralized framework) is a particular instance of the `h_efficient` system (our hallucination system) where the inference loops and memory module is adjusted.
3. Both `h_efficient` and `h_vanilla` are our hallucination system with slight differences in implementation, refer to these links for more details:
    - [`h_efficient`](https://github.com/KevinBian107/RPLH/tree/master/rplh/h_efficient)
    - [`h_vanilla`](https://github.com/KevinBian107/RPLH/tree/master/rplh/h_vanilla)
4. We have created inference functions in each of the system to be called seperately, or an direct interface can be called by specifying args on the root folder level of rplh (`inference.py`).

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
python rplh/test.py --module_name "h_efficient" --model_name "qwen2.5:14b-instruct-q3_K_L" --num_trials 5 --box_num_upper_bound 1 --start_iter 1
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