import json
import os
import numpy as np
from collections import defaultdict
from typing import List, Dict


def load_and_combine_data(file_path: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Load and combine conversations from train, dev, and test sets in the JSON file.

    Args:
        file_path (str): Path to the raw JSON file.

    Returns:
        Dict[str, List[Dict[str, str]]]: Combined conversations with unique IDs formatted as required.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    conversation_id = 0
    conversations = {}

    # Combine conversations from train, dev, and test sets
    for split in ['train', 'dev', 'test']:
        for conversation in data[split]:
            formatted_conversation = []
            for turn in conversation['original']:
                formatted_conversation.append({
                    "role": turn[0],
                    "content": turn[1]
                })
            conversations[str(conversation_id)] = formatted_conversation
            conversation_id += 1

    return conversations


def calculate_statistics(data: List[int]) -> Dict[str, float]:
    """
    Calculate various statistics on a list of numbers and convert NumPy types to native Python types.

    Args:
        data (List[int]): List of numerical values.

    Returns:
        Dict[str, float]: Dictionary containing calculated statistics.
    """
    return {
        "average": float(np.mean(data)),
        "maximum": int(np.max(data)),
        "minimum": int(np.min(data)),
        "median": float(np.median(data)),
        "variance": float(np.var(data, ddof=1)),  # Use ddof=1 for sample variance
        "std_dev": float(np.std(data, ddof=1))   # Use ddof=1 for sample standard deviation
    }



def compute_role_statistics(conversations: Dict[str, List[Dict[str, str]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute statistics for each role in the conversations.

    Args:
        conversations (Dict[str, List[Dict[str, str]]]): The combined conversations data.

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: Dictionary containing word and character statistics for each role.
    """
    role_stats = {
        "agent": {"words": [], "characters": []},
        "customer": {"words": [], "characters": []},
        "action": {"words": [], "characters": []}
    }

    for conversation in conversations.values():
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            word_count = len(content.split())
            char_count = len(content)

            if role in role_stats:
                role_stats[role]["words"].append(word_count)
                role_stats[role]["characters"].append(char_count)

    # Calculate statistics for each role
    stats = {}
    for role, metrics in role_stats.items():
        stats[role] = {
            "words": calculate_statistics(metrics["words"]),
            "characters": calculate_statistics(metrics["characters"])
        }

    return stats


def compute_dialogue_statistics(conversations: Dict[str, List[Dict[str, str]]]) -> Dict[str, float]:
    """
    Calculate statistics for the length of each dialogue in terms of word count.

    Args:
        conversations (Dict[str, List[Dict[str, str]]]): The combined conversations data.

    Returns:
        Dict[str, float]: Statistics on the lengths of dialogues.
    """
    dialogue_lengths = [sum(len(turn["content"].split()) for turn in dialogue) for dialogue in conversations.values()]
    
    return calculate_statistics(dialogue_lengths)


def save_statistics(conversations: Dict[str, List[Dict[str, str]]], output_dir: str) -> None:
    """
    Save detailed statistics about the dataset including dialogue and role-based statistics.

    Args:
        conversations (Dict[str, List[Dict[str, str]]]): The combined conversations data.
        output_dir (str): Directory to save the statistics.
    """
    dialogue_statistics = compute_dialogue_statistics(conversations)
    role_statistics = compute_role_statistics(conversations)

    all_stats = {
        "dialogue_statistics": dialogue_statistics,
        "role_statistics": role_statistics
    }

    stats_path = os.path.join(output_dir, 'statistics.json')
    with open(stats_path, 'w') as stats_file:
        json.dump(all_stats, stats_file, indent=4)
    
    print(f"Statistics saved to {stats_path}")


def save_processed_data(conversations: Dict[str, List[Dict[str, str]]], output_dir: str) -> None:
    """
    Save processed conversations data.

    Args:
        conversations (Dict[str, List[Dict[str, str]]]): The combined conversations data.
        output_dir (str): Directory to save the processed data.
    """
    processed_data_path = os.path.join(output_dir, 'processed_formatted_conversations.json')
    with open(processed_data_path, 'w') as processed_file:
        json.dump(conversations, processed_file, indent=4)
    
    print(f"Processed data saved to {processed_data_path}")


def main():
    # File paths
    input_file_path = os.path.join("data" , "abcd_v1.1.json")  # Update with your input JSON file path
    output_dir = 'data'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load and process data
    conversations = load_and_combine_data(input_file_path)

    # Save statistics
    save_statistics(conversations, output_dir)

    # Save processed data
    save_processed_data(conversations, output_dir)


if __name__ == "__main__":
    main()
