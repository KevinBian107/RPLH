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

def get_state_spy_for_analysis(base_dir: str):
    '''Main retrieval fiunction for evaluations'''
    
    trial_dirs = [f"trial_{i}" for i in range(1, 11)]
    trial_data = []
    trial_agent_counts = {}
    for trial in trial_dirs:
        trial_path = os.path.join(base_dir, trial, "env_pg_state_3_3/pg_state0", "_w_no_history_gpt-4o-mini")
        pg_state_file = os.path.join(trial_path, "pg_state/pg_state0.json")
        response_dir = os.path.join(trial_path, "response")
        spy_model_dir = os.path.join(trial_path, "spy_model")
        state_dir = os.path.join(trial_path, "pg_state")
        
        # Success or not
        state_files = os.listdir(state_dir)
        state_files.sort(key=lambda x: int(x.split(".")[0][8:]))
        last_file = state_files[-1]
        state_path = os.path.join(state_dir, last_file)
        # print (trial, last_file)
        
        with open(state_path, "r") as f:
            try:
                response = json.load(f)
                if not all([len(value) == 0 or len(value) == 2 for value in response.values()]):
                    print(f"{trial}: Not Converged")
                    print(response)
            except json.JSONDecodeError:
                print(f"Invalid JSON in {state_path} of {trial}")
                continue
        
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
        
        # Read response JSON files
        response_files = [f for f in os.listdir(response_dir) if f.endswith(".json")]
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
        
        # Start spy_model extraction
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
        
        trial_agent_counts[trial] = trial_counts  # Save the trial-specific Counter
        # print(trial, trial_counts)

    # Create a dataframe from trial_agent_counts
    agent_names = set(agent for counts in trial_agent_counts.values() for agent in counts)
    agent_df = pd.DataFrame.from_dict(
        {trial: {agent: trial_agent_counts.get(trial, {}).get(agent, 0) for agent in agent_names} for trial in trial_dirs},
        orient="index"
    ).fillna(0).astype(int)
    agent_df.index.name = "Trial"

    df = pd.DataFrame(trial_data)
    df["Avg_Boxes_To_Targets_Per_Response"] = df["Boxes_To_Targets"] / df["Num_Responses"]
    df["Avg_Boxes_To_Other_Per_Response"] = df["Boxes_To_Other"] / df["Num_Responses"]
    
    return df, agent_df


def get_embedding(text: str, url="http://localhost:11434/api/embeddings") -> np.ndarray:
    """
    Get text embeddings from local LLM.

    Args:
        text (str): The text for generating the embeddings.
        url (str): The API endpoint URL.

    Returns:
        The embeddings as a numpy array.
    """
    data = {
        "model": 'nomic-embed-text', 
        "prompt": text
    }
    try:
        response = requests.post(url=url, json=data)

        if response.status_code == 200:
            response_emb = np.array(response.json()['embedding'])
            return response_emb
        else:
            print("Error:", response.status_code, response.json())
            return None
    except Exception as e:
        print(f"API call failed: {e}")
        return None

def cosine_sim(vec1: np.array, vec2: np.array) -> float:
    """
    Computes the cosine similarity between two vectors.

    Args:
        vec1: The first vector.
        vec2: The second vector.

    Returns:
        The cosine similarity between the two vectors.
    """
    # Dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Magnitude
    magnitude_vec1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude_vec2 = sum(b ** 2 for b in vec2) ** 0.5
    
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    
    return dot_product / (magnitude_vec1 * magnitude_vec2)

def pearson_correlation(vec1, vec2):
    """
    Computes the Pearson correlation coefficient between two vectors.

    Args:
        vec1: The first vector.
        vec2: The second vector.

    Returns:
        The Pearson correlation coefficient between the two vectors.
    """
    mean1, mean2 = sum(vec1) / len(vec1), sum(vec2) / len(vec2)
    numerator = sum((a - mean1) * (b - mean2) for a, b in zip(vec1, vec2))
    denominator = (sum((a - mean1) ** 2 for a in vec1) ** 0.5) * (sum((b - mean2) ** 2 for b in vec2) ** 0.5)
    if denominator == 0:
        return 0
    
    return numerator / denominator