import json
import numpy as np
from typing import Dict , Union , Callable
import os
import time
import functools

def extract_utterances(data: dict, *roles) -> dict:
    """Extract utterances for a specific role from each conversation."""
    return {
        key: [utt for utt in conv if utt["role"] in roles ]
        for key, conv in data.items()
    }

def split_data(coff: float, data: dict):
    """Split data into train and test sets according to a coefficient."""
    train_data_source = {
        str(key): data[key]
        for key in list(data.keys())[ : int(coff * len(data))]
    }
    test_data_source = {
        str(key): data[key]
        for key in list(data.keys())[ int(coff * len(data)) : len(data)]
    }
    return train_data_source, test_data_source


def read_json_file(file_path) : 
    try : 
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e : 
        print(f"error occured in read_json_file  : {e}")


def get_random_n_pairs(input_dict, n):
    """
    Get n random key-value pairs from a dictionary.

    Parameters:
    input_dict (dict): The input dictionary.
    n (int): The number of key-value pairs to return.

    Returns:
    dict: A dictionary containing n random key-value pairs.
    """
    if n <= 0:
        return {}

    if n >= len(input_dict):
        return input_dict

    # Get n random keys from the dictionary
    random_keys = list(input_dict.keys())[:n]

    # Create a new dictionary with the random keys and their corresponding values
    random_pairs = {key: input_dict[key] for key in random_keys}

    return random_pairs


def convert_numpy_int_to_python(data):
    if isinstance(data, dict):
        return {key: convert_numpy_int_to_python(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_int_to_python(element) for element in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    else:
        return data


def save_dict_to_json(data: dict, file_path: str) -> None:
    """
    Deletes the existing JSON file (if it exists) and saves a dictionary to a new JSON file,
    converting non-serializable types like NumPy arrays to lists.
    
    Args:
        data (dict): The dictionary to save.
        file_path (str): The path to the JSON file.
        
    Returns:
        None
    """
    # Step 1: Delete the existing JSON file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted existing file: {file_path}")

    # Step 2: Save the new data to the JSON file
    def convert_to_serializable(obj):
        # Handle NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Add handling for other non-serializable types here if needed
        raise TypeError(f"Type {type(obj)} is not JSON serializable")
    
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4, default=convert_to_serializable)
        print(f"Dictionary successfully saved to {file_path}")
    except Exception as e:
        raise IOError(f"An error occurred while saving the dictionary to {file_path}: {e}")


def apply_str_filter(s) : 
    return [''.join([w[0].upper() for w in sentence.split(" ")])  for sentence in s   ]

def save_transition_matrix(transition_matrix, filename='transition_matrix_ada.npy'):
    np.save(filename, transition_matrix)
def read_transition_matrix(filename='transition_matrix_ada.npy'):
    return np.load(filename)



def pre_process_llm_output(function_calling_message: str) -> Dict[str, Union[str, Dict]]:
    start_idx = function_calling_message.find('{')
    end_idx = function_calling_message.rfind('}')
    
    try:
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            # Extract the JSON-like string part
            json_str = function_calling_message[start_idx:end_idx + 1]
            
            # Perform replacement of single quotes only if they are not inside double quotes
            inside_quotes = False
            processed_str = []
            for char in json_str:
                if char == '"':  # Toggle inside_quotes flag
                    inside_quotes = not inside_quotes
                if char == "'" and not inside_quotes:
                    processed_str.append('"')
                else:
                    processed_str.append(char)
            
            processed_json_str = ''.join(processed_str)
            
            # Attempt to load the processed string as JSON
            func_call = json.loads(processed_json_str)
            return func_call
    except Exception as e:
        print(f"Error: {e}")
        return function_calling_message
    


def read_json_to_dict(file_path: str) -> dict:
    """
    Reads a JSON file and loads its contents into a dictionary.
    
    Args:
        file_path (str): The path to the JSON file.
        
    Returns:
        dict: The contents of the JSON file as a Python dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        return {}



def exponential_backoff(retries: int = 5, backoff_in_seconds: int = 1, max_backoff: int = 32):
    """
    A decorator to add exponential backoff to a function in case of exceptions.
    
    Args:
        retries (int): Maximum number of retries before giving up.
        backoff_in_seconds (int): Initial backoff duration in seconds.
        max_backoff (int): Maximum allowed backoff time in seconds.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            backoff = backoff_in_seconds
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt == retries:
                        raise e  # Raise the exception if max retries are exceeded
                    time.sleep(min(backoff, max_backoff))  # Wait with capped exponential backoff
                    backoff *= 2
        return wrapper
    return decorator
