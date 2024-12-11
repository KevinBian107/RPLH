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

def calculate_distance(grid_data):
    '''Parsing grid data to extract coordinates and objects'''
    positions = {}
    for coord, items in grid_data.items():
        x, y = map(float, coord.split("_"))
        for item in items:
            kind, color = item.split("_")
            if color not in positions:
                positions[color] = {"box": [], "target": []}
            positions[color][kind].append((x, y))

    all_distances = {}
    for color, pos in positions.items():
        boxes = np.array(pos["box"])
        targets = np.array(pos["target"])
        
        if boxes.size > 0 and targets.size > 0:
            distances = np.linalg.norm(boxes[:, None] - targets, axis=2, ord=1)  # 1-norm
            min_1_norm = distances.min()
            
            distances_2 = np.linalg.norm(boxes[:, None] - targets, axis=2, ord=2)  # 2-norm
            average_2_norm = distances_2.mean()
            
            all_distances[color] = {"1-norm": min_1_norm, "2-norm": average_2_norm}
        else:
            all_distances[color] = {"1-norm": None, "2-norm": None}
    
    df = pd.DataFrame(all_distances).T
    if df.empty:
        return (0, 0)
    
    df.columns = ["1-Norm Smallest Distance", "2-Norm Average Distance"]
    df['1-Norm Smallest Distance'] = df['1-Norm Smallest Distance'] + 1 # same grid need one more stepfor 1 norm
    
    return tuple(df.sum(axis=0))

def get_response_data(response_dir, trial_data, trial, num_boxes, num_targets, consectutive_syntactic_error_limit=3):
    '''Get all resposne data'''
    response_files = os.listdir(response_dir)
    valid_responses = 0
    boxes_to_targets = 0
    boxes_to_other = 0
    
    sythetic_count = 0
    for response_file in response_files:
        response_path = os.path.join(response_dir, response_file)
        with open(response_path, "r") as f:
            try:
                response = json.load(f)
            except json.JSONDecodeError:
                print(f"Invalid JSON in {response_file} of {trial}")
                continue 
            
        if response == "Syntactic Error":
            sythetic_count+=1
            # can't read() first
            response = {}
            
            if sythetic_count > consectutive_syntactic_error_limit:
                # over consectutive_syntactic_error_limit, break directly
                print(f"{trial}: Not converge")
                break
        
        sythetic_count = 0
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
    "Num_Responses": valid_responses,
    "Boxes_To_Targets": boxes_to_targets,
    "Boxes_To_Other": boxes_to_other,
    })
        
    return trial_data

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
                # print(response)
                success_or_not[trial] = "Not Converged"
            else:
                success_or_not[trial] = "Converged"
                
        except json.JSONDecodeError:
            print(f"Invalid JSON in {state_path} of {trial}")
        
        return success_or_not

def get_spy_count_data(spy_model_dir, trial_spy_counts, trial):
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

def get_agent_model(agent_dir, trial_agent_model, trial):
    '''Get all agent model data'''
    trial_agent_descriptions = {}
    if os.path.exists(agent_dir):
        agent_files = os.listdir(agent_dir)
        
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

def get_justification(justification_dir, trial_justification, trial):
    '''Get all justification data'''
    trial_agent_descriptions = []
    if os.path.exists(justification_dir):
        just_files = os.listdir(justification_dir)
        
        for just_file in just_files:
            just_path = os.path.join(justification_dir, just_file)
            with open(just_path, "r") as f:
                description = f.read()

            trial_agent_descriptions.append(description)
    
    trial_justification[trial] = trial_agent_descriptions
    return trial_justification

def get_current_distance(state_dir, trial, cur_distance):
    "Check current 1 norm and 2 norm distance"
    
    state_path = os.listdir(state_dir)
    state_path.sort(key=lambda x: int(x.split(".")[0][8:]))
    
    all_distance = []
    for state_file in state_path:
        file = os.path.join(state_dir, state_file)
        with open(file, "r") as f:
            try:
                response = json.load(f)
                distance = calculate_distance(response)
                all_distance.append(distance)
                
            except json.JSONDecodeError:
                print(f"Invalid JSON in {file} of {trial}")
    
    cur_distance[trial] = all_distance 
    return cur_distance


def get_data(base_dir, trial_num, demo=False):
    '''Main retrieval fiunction for evaluations, all tensor operations'''
    
    trial_dirs = [f"trial_{i}" for i in range(1, trial_num+1)]
    trial_data = []
    trial_spy_counts = {}
    success_or_not = {}
    trial_agent_model = {}
    trial_spy_model = {}
    trial_justification = {}
    cur_distance = {}
    for trial in trial_dirs:
        trial_path = os.path.join(base_dir, trial, "env_pg_state_3_3/pg_state0", "_w_no_history_gpt-4o-mini")
        
        if demo:
            trial_path = base_dir
            print(base_dir)
            
        pg_state_file = os.path.join(trial_path, "pg_state/pg_state0.json")
        response_dir = os.path.join(trial_path, "response")
        spy_model_dir = os.path.join(trial_path, "spy_model")
        state_dir = os.path.join(trial_path, "pg_state")
        agent_dir = os.path.join(trial_path, "agent_model")
        justification_dir = os.path.join(trial_path, "justification")
        
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
        cur_distance = get_current_distance(state_dir, trial, cur_distance)
        trial_data = get_response_data(response_dir, trial_data, trial, num_boxes, num_targets)
        trial_spy_counts = get_spy_count_data(spy_model_dir, trial_spy_counts, trial)
        trial_agent_model = get_agent_model(agent_dir, trial_agent_model, trial)
        trial_spy_model = get_agent_model(spy_model_dir, trial_spy_model, trial)
        trial_justifications = get_justification(justification_dir, trial_justification, trial)
    
    df = pd.DataFrame(trial_data)
    df["Avg_Boxes_To_Targets_Per_Response"] = df["Boxes_To_Targets"] / df["Num_Responses"]
    df["Avg_Boxes_To_Other_Per_Response"] = df["Boxes_To_Other"] / df["Num_Responses"]
    
    success_df = pd.DataFrame(trial_success.items(), columns=["Trial", "Convergence"])
    
    dist_df = pd.DataFrame(cur_distance.items(), columns=["Trial", "Distance"])
    
    agent_names = set(agent for counts in trial_spy_counts.values() for agent in counts)
    spy_count_df = pd.DataFrame.from_dict(
        {trial: {agent: trial_spy_counts.get(trial, {}).get(agent, 0) for agent in agent_names} for trial in trial_dirs},
        orient="index"
    ).fillna(0).astype(int)
    spy_count_df.index.name = "Trial"
    
    agent_names = set(agent for counts in trial_spy_model.values() for agent in counts)
    spy_df = pd.DataFrame.from_dict(
        {trial: {agent: trial_spy_model.get(trial, {}).get(agent, 0) for agent in agent_names} for trial in trial_dirs},
        orient="index"
    )
    spy_df.index.name = "Trial"
    
    agent_names = set(agent for counts in trial_agent_model.values() for agent in counts)
    att_df = pd.DataFrame.from_dict(
        {trial: {agent: trial_agent_model.get(trial, {}).get(agent, 0) for agent in agent_names} for trial in trial_dirs},
        orient="index"
    )
    att_df.index.name = "Trial"
    
    justification_df = pd.DataFrame(trial_justifications.items(), columns=["Trial", "Justifications"])
    
    return df, success_df, spy_count_df, spy_df, att_df, justification_df, dist_df