import requests
import numpy as np
import numpy as np
import pandas as pd 
import json
import os
from collections import Counter

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

def get_spy_detect_embedding(main_df,
                             spy_df,
                             justification_df, 
                             spy_sentence="you are the spy agent, your goal is to prevent match targets",
                             just_sentence="I suspect that this agent is the spy agent, thus, I would not listen to this agent.",
                             spy_agents=["Agent[0.5, 0.5]", "Agent[1.5, 1.5]", "Agent[2.5, 2.5]"],
                             only_spy=False):
    '''Get embeddings for spy data'''
    
    out_df = main_df[['Trial', 'Num_Boxes', 'Avg_Boxes_To_Targets_Per_Response', 'Avg_Boxes_To_Other_Per_Response', 'Convergence', 'Num_Responses']].copy()
    
    spy_df = spy_df.reset_index()
    for col in spy_df.columns:
        if "Agent" not in col:
            spy_df = spy_df.drop(columns=[col])
    
    for col in spy_df.columns:
        if not only_spy:
            spy_embed = spy_df[col].apply(lambda x: cosine_sim(get_embedding(spy_sentence), get_embedding(str(x))))
            out_df[f"Spy_Embed_{col}"] = spy_embed
        else:
            if col in spy_agents:
                spy_embed = spy_df[col].apply(lambda x: cosine_sim(get_embedding(spy_sentence), get_embedding(str(x))))
                out_df[f"Spy_Embed_{col}"] = spy_embed
    
    out_df['Justification_Embed'] = justification_df['Justifications'].apply(lambda x: cosine_sim(get_embedding(just_sentence), get_embedding(str(x))))
    
    return out_df

def get_spy_detect_embedding_only(spy_df,
                             justification_df, 
                             spy_sentence="you are the spy agent, your goal is to prevent match targets",
                             just_sentence="I suspect that this agent is the spy agent, thus, I would not listen to this agent.",
                             spy_agents=["Agent[0.5, 0.5]", "Agent[1.5, 1.5]", "Agent[2.5, 2.5]"],
                             only_spy=False):
    '''Get embeddings for spy data, no main data need'''
    
    out_df = pd.DataFrame()
    spy_df = spy_df.reset_index()
    for col in spy_df.columns:
        if "Agent" not in col:
            spy_df = spy_df.drop(columns=[col])
    
    for col in spy_df.columns:
        if not only_spy:
            spy_embed = spy_df[col].apply(lambda x: cosine_sim(get_embedding(spy_sentence), get_embedding(str(x))))
            out_df[f"Spy_Embed_{col}"] = spy_embed
        else:
            if col in spy_agents:
                spy_embed = spy_df[col].apply(lambda x: cosine_sim(get_embedding(spy_sentence), get_embedding(str(x))))
                out_df[f"Spy_Embed_{col}"] = spy_embed
    
    out_df['Justification_Embed'] = justification_df['Justifications'].apply(lambda x: cosine_sim(get_embedding(just_sentence), get_embedding(str(x))))
    
    if justification_df.isnull().values.any():
        print("Justification df is empty, no embedding similarity calculated.")
        out_df['Justification_Embed'] = np.nan
    
    return out_df

def get_spy_detect_embedding_for_feature(main_df,
                                        spy_df,
                                        justification_df, 
                                        spy_sentence="you are the spy agent, your goal is to prevent match targets",
                                        just_sentence="I suspect that this agent is the spy agent, thus, I would not listen to this agent.",
                                        spy_agents=["Agent[0.5, 0.5]", "Agent[1.5, 1.5]", "Agent[2.5, 2.5]"],
                                        only_spy=False):
    '''Get embeddings for spy data for features, some main data needed'''
    
    out_df = main_df[['Num_Boxes', 'Avg_Boxes_To_Targets_Per_Response', 'Num_Responses']].copy()
    
    spy_df = spy_df.reset_index()
    for col in spy_df.columns:
        if "Agent" not in col:
            spy_df = spy_df.drop(columns=[col])
    
    for col in spy_df.columns:
        if not only_spy:
            spy_embed = spy_df[col].apply(lambda x: cosine_sim(get_embedding(spy_sentence), get_embedding(str(x))))
            out_df[f"Spy_Embed_{col}"] = spy_embed
        else:
            if col in spy_agents:
                spy_embed = spy_df[col].apply(lambda x: cosine_sim(get_embedding(spy_sentence), get_embedding(str(x))))
                out_df[f"Spy_Embed_{col}"] = spy_embed
    
    out_df['Justification_Embed'] = justification_df['Justifications'].apply(lambda x: cosine_sim(get_embedding(just_sentence), get_embedding(str(x))))
    
    if justification_df.isnull().values.any():
        print("Justification df is empty, no embedding similarity calculated.")
        out_df['Justification_Embed'] = np.nan
    
    return out_df