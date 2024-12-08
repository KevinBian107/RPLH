import requests
import numpy as np
import numpy as np
import pandas as pd 
import json
import os
from collections import Counter

def count_boxes_and_targets(pg_state):
    '''Function to count boxes and targets in a given pg_state'''
    num_boxes = sum(1 for items in pg_state.values() for item in items if item.startswith("box_"))
    num_targets = sum(1 for items in pg_state.values() for item in items if item.startswith("target_"))
    return num_boxes, num_targets

def get_response_data(response_dir, trial_data, trial, num_boxes, num_targets):
    '''Get all resposne data'''
    response_files = os.listdir(response_dir)
    valid_responses = 0
    boxes_to_targets = 0
    boxes_to_other = 0

    for response_file in response_files:
        response_path = os.path.join(response_dir, response_file)
        with open(response_path, "r") as f:
            try:
                response = json.load(f)
            except json.JSONDecodeError:
                print(f"Invalid JSON in {response_file} of {trial}")
                continue 
            
        if response == "Syntactic Error":
            # can't read() first
            continue
        
        valid_responses += 1
        for action in response.values():
            if "move(" in action:
                parts = action.split(", ")
                if "target" in parts[-1]:
                    boxes_to_targets += 1
                else:
                    boxes_to_other += 1
        
        # Store trial data
        trial_data.append({
        "Trial": trial,
        "Num_Boxes": num_boxes,
        "Num_Targets": num_targets,
        "Num_Responses": valid_responses,
        "Boxes_To_Targets": boxes_to_targets,
        "Boxes_To_Other": boxes_to_other,
    })
        
        return trial_data

def get_spy_model_data(spy_model_dir, trial_spy_counts, trial):
    '''Use counter to store spy model data'''
    trial_counts = Counter()  # Create a separate Counter for this trial
    if os.path.exists(spy_model_dir):
        spy_model_files = [f for f in os.listdir(spy_model_dir) if f.endswith(".json")]
        
        for spy_model_file in spy_model_files:
            spy_model_path = os.path.join(spy_model_dir, spy_model_file)
            with open(spy_model_path, "r") as f:
                try:
                    spy_model = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error decoding {spy_model_file} in {trial}: {e}")
                    continue

            # Extract agent mentions and count them
            agents = list(spy_model.keys())
            trial_counts.update(agents)
    
    trial_spy_counts[trial] = trial_counts  # Save the trial-specific Counter
    
    return trial_spy_counts

def get_convergence_data(state_dir, trial, success_or_not):
    "Check converge or not"
    state_files = os.listdir(state_dir)
    state_files.sort(key=lambda x: int(x.split(".")[0][8:]))
    last_file = state_files[-1]
    state_path = os.path.join(state_dir, last_file)
    
    with open(state_path, "r") as f:
        try:
            response = json.load(f)
            
            # catch case for two API responses
            if not all([len(value) == 0 or len(value) == 2 for value in response.values()]):
                print(f"{trial}: Not Converged")
                print(response)
                success_or_not[trial] = "Not Converged"
            else:
                success_or_not[trial] = "Converged"
                
        except json.JSONDecodeError:
            print(f"Invalid JSON in {state_path} of {trial}")
        
        return success_or_not

def get_agent_model(agent_dir, trial_agent_model, trial):
    '''Get all agent model data'''
    trial_agent_descriptions = {}
    if os.path.exists(agent_dir):
        agent_files = [f for f in os.listdir(agent_dir) if f.endswith(".json")]
        
        for agent_file in agent_files:
            agent_path = os.path.join(agent_dir, agent_file)
            with open(agent_path, "r") as f:
                try:
                    agent_model = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error decoding {agent_file} in {trial}: {e}")
                    continue

            # Extract agent descriptions and store them by agent name
            for agent, description in agent_model.items():
                if agent not in trial_agent_descriptions:
                    trial_agent_descriptions[agent] = []
                trial_agent_descriptions[agent].append(description)
    
    trial_agent_model[trial] = trial_agent_descriptions
    return trial_agent_model

def get_data(base_dir, trial_num):
    '''Main retrieval fiunction for evaluations'''
    
    trial_dirs = [f"trial_{i}" for i in range(1, trial_num+1)]
    trial_data = []
    trial_spy_counts = {}
    success_or_not = {}
    trial_agent_model = {}
    trial_spy_model = {}
    for trial in trial_dirs:
        trial_path = os.path.join(base_dir, trial, "env_pg_state_3_3/pg_state0", "_w_no_history_gpt-4o-mini")
        pg_state_file = os.path.join(trial_path, "pg_state/pg_state0.json")
        response_dir = os.path.join(trial_path, "response")
        spy_model_dir = os.path.join(trial_path, "spy_model")
        state_dir = os.path.join(trial_path, "pg_state")
        agent_dir = os.path.join(trial_path, "agent_model")
        
        # Skip trial if pg_state file is missing
        if not os.path.exists(pg_state_file):
            print(f"pg_state0.json not found for {trial}")
            continue

        # Read pg_state0.json and count boxes and targets
        with open(pg_state_file, "r") as f:
            pg_state = json.load(f)
        num_boxes, num_targets = count_boxes_and_targets(pg_state)

        # Skip trial if response directory is missing
        if not os.path.exists(response_dir):
            print(f"Response directory not found for {trial}")
            continue
        
        trial_success = get_convergence_data(state_dir, trial, success_or_not)
        trial_data = get_response_data(response_dir, trial_data, trial, num_boxes, num_targets)
        trial_spy_counts = get_spy_model_data(spy_model_dir, trial_spy_counts, trial)
        trial_agent_model = get_agent_model(agent_dir, trial_agent_model, trial)
        trial_spy_model = get_agent_model(spy_model_dir, trial_spy_model, trial)
    
    success_df = pd.DataFrame(trial_success.items(), columns=["Trial", "Convergence"])
    
    agent_names = set(agent for counts in trial_spy_counts.values() for agent in counts)
    agent_df = pd.DataFrame.from_dict(
        {trial: {agent: trial_spy_counts.get(trial, {}).get(agent, 0) for agent in agent_names} for trial in trial_dirs},
        orient="index"
    ).fillna(0).astype(int)
    agent_df.index.name = "Trial"
    
    agent_names = set(agent for counts in trial_agent_model.values() for agent in counts)
    att_df = pd.DataFrame.from_dict(
        {trial: {agent: trial_agent_model.get(trial, {}).get(agent, 0) for agent in agent_names} for trial in trial_dirs},
        orient="index"
    )
    agent_df.index.name = "Trial"
    
    agent_names = set(agent for counts in trial_spy_model.values() for agent in counts)
    spy_df = pd.DataFrame.from_dict(
        {trial: {agent: trial_spy_model.get(trial, {}).get(agent, 0) for agent in agent_names} for trial in trial_dirs},
        orient="index"
    )
    agent_df.index.name = "Trial"

    df = pd.DataFrame(trial_data)
    df["Avg_Boxes_To_Targets_Per_Response"] = df["Boxes_To_Targets"] / df["Num_Responses"]
    df["Avg_Boxes_To_Other_Per_Response"] = df["Boxes_To_Other"] / df["Num_Responses"]
    
    return df, agent_df, success_df, att_df, spy_df