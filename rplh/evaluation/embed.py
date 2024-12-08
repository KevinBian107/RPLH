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