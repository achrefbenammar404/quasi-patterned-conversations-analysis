import json
import pandas as pd
import numpy as np
from typing import List, Dict

class ConversationDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.conversations = {}
        self.formatted_conversations = {}

    def load_data(self) -> None:
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        
        # Extract conversations from train, dev, and test data
        conversation_id = 0
        for conversation_set in ["train", "dev", "test"]:
            for conversation in data[conversation_set]:
                self.conversations[conversation_id] = conversation['original']
                conversation_id += 1
    
    def pre_process_dialogue(self, dialogue: List[List[str]]) -> List[Dict[str, str]]:
        formatted_conversation = []
        for turn in dialogue:
            formatted_conversation.append({"role": turn[0], "content": turn[1]})
        return formatted_conversation
    
    def process_conversations(self) -> None:
        for conversation_id, dialogue in self.conversations.items():
            self.formatted_conversations[conversation_id] = self.pre_process_dialogue(dialogue)

    def save_formatted_data(self, output_file: str) -> None:
        with open(output_file, 'w') as json_file:
            json.dump(self.formatted_conversations, json_file, indent=4)

    def analyze_conversations(self) -> Dict[str, Dict[str, float]]:
        role_word_counts_per_conversation = {}
        role_char_counts_per_conversation = {}

        # Calculate word and character statistics for each role
        for dialogue in self.formatted_conversations.values():
            for turn in dialogue:
                role = turn['role']
                words = len(turn['content'].split())
                chars = len(turn['content'])

                if role not in role_word_counts_per_conversation:
                    role_word_counts_per_conversation[role] = []
                if role not in role_char_counts_per_conversation:
                    role_char_counts_per_conversation[role] = []

                role_word_counts_per_conversation[role].append(words)
                role_char_counts_per_conversation[role].append(chars)

        role_stats = {}
        for role in role_word_counts_per_conversation:
            word_counts = role_word_counts_per_conversation[role]
            char_counts = role_char_counts_per_conversation[role]

            role_stats[role] = {
                "average_words": np.mean(word_counts),
                "max_words": np.max(word_counts),
                "min_words": np.min(word_counts),
                "median_words": np.median(word_counts),
                "variance_words": np.var(word_counts),
                "std_dev_words": np.std(word_counts),
                "average_chars": np.mean(char_counts),
                "max_chars": np.max(char_counts),
                "min_chars": np.min(char_counts),
                "median_chars": np.median(char_counts),
                "variance_chars": np.var(char_counts),
                "std_dev_chars": np.std(char_counts)
            }
        
        return role_stats

    def save_statistics(self, stats: Dict[str, Dict[str, float]], output_file: str) -> None:
        with open(output_file, 'w') as file:
            for role, role_stats in stats.items():
                file.write(f"Role: {role}\n")
                file.write(f"  Average words per conversation: {role_stats['average_words']:.2f}\n")
                file.write(f"  Maximum words per conversation: {role_stats['max_words']}\n")
                file.write(f"  Minimum words per conversation: {role_stats['min_words']}\n")
                file.write(f"  Median words per conversation: {role_stats['median_words']:.2f}\n")
                file.write(f"  Variance of words per conversation: {role_stats['variance_words']:.2f}\n")
                file.write(f"  Standard deviation of words per conversation: {role_stats['std_dev_words']:.2f}\n")
                file.write(f"  Average characters per conversation: {role_stats['average_chars']:.2f}\n")
                file.write(f"  Maximum characters per conversation: {role_stats['max_chars']}\n")
                file.write(f"  Minimum characters per conversation: {role_stats['min_chars']}\n")
                file.write(f"  Median characters per conversation: {role_stats['median_chars']:.2f}\n")
                file.write(f"  Variance of characters per conversation: {role_stats['variance_chars']:.2f}\n")
                file.write(f"  Standard deviation of characters per conversation: {role_stats['std_dev_chars']:.2f}\n")
                file.write("\n")

if __name__ == "__main__":
    # Specify file paths
    input_file_path = 'src/data/abcd_v1.1.json'
    output_formatted_file = 'src/data/processed_formatted_conversations.json'
    output_stats_file = 'src/data/conversation_statistics.txt'

    # Create the data loader object
    loader = ConversationDataLoader(file_path=input_file_path)

    # Load, process, and save formatted conversations
    loader.load_data()
    loader.process_conversations()
    loader.save_formatted_data(output_file=output_formatted_file)

    # Perform analysis and save statistics
    stats = loader.analyze_conversations()
    loader.save_statistics(stats=stats, output_file=output_stats_file)