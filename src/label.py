from typing import Dict, List, Any
from src.llm import CollectionLLM
from src.utils.utils import pre_process_llm_output
import json
from collections import Counter
from typing import Dict, List
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_md")

class Label:

    def generate_cluster_by_intent(intent_by_cluster: Dict[str, str]) -> Dict[str, str]:
        """
        Generates a dictionary that maps each intent to its corresponding cluster.

        Args:
            intent_by_cluster (Dict[str, str]): Dictionary mapping cluster identifiers to intents.

        Returns:
            Dict[str, str]: Dictionary mapping intents to cluster identifiers.
        """
        return {value: key for key, value in intent_by_cluster.items()}


    def label_clusters_by_verbphrases(closest_utterances: Dict[int, List[str]] ) -> Dict[str, str]:
        """
        Labels clusters by analyzing the closest utterances and extracting the most frequent verb phrases.

        Args:
            closest_utterances (Dict[int, List[str]]): Dictionary mapping cluster indices to lists of closest utterances.
            model (str): The model name to be used from the CollectionLLM.

        Returns:
            Dict[str, str]: Dictionary mapping cluster indices to their most frequent verb phrases.
        """
        intent_by_cluster = {}

        for cluster, utterances in closest_utterances.items():
            all_verb_phrases = []

            # Extract verb phrases from each utterance
            for utterance in utterances:
                verb_phrases = Label.extract_verb_phrases(utterance)
                all_verb_phrases.extend(verb_phrases)

            # Count the frequency of each verb phrase
            verb_phrase_counter = Counter(all_verb_phrases)

            # Identify the most common verb phrase
            if verb_phrase_counter:
                most_common_verb_phrase, _ = verb_phrase_counter.most_common(1)[0]
                intent_by_cluster[str(cluster)] = most_common_verb_phrase
            else:
                # In case no verb phrases are found
                intent_by_cluster[str(cluster)] = "No verb phrase detected"

        return intent_by_cluster
    def label_clusters_by_closest_utterances(closest_utterances: Dict[int, List[str]], 
                                             model: str) -> Dict[str, str]:
        """
        Labels clusters by analyzing the closest utterances and generating intents using an LLM model.

        Args:
            closest_utterances (Dict[int, List[str]]): Dictionary mapping cluster indices to lists of closest utterances.
            model (str): The model name to be used from the CollectionLLM.

        Returns:
            Dict[str, str]: Dictionary mapping cluster indices to their identified intents.
        """
        try:
            client = CollectionLLM.llm_collection[model]
        except Exception as e:
            print(f"Error in label_clusters_by_closest_utterances, model is not included in CollectionLLM: {e}")
            return {}

        intent_by_cluster = {}
        for cluster, utterances in closest_utterances.items():
            prompt = (
                "You will be given some utterances from a conversation between a customer support agent and clients from a specific company. "
                "You need to extract the intent of these utterances. Your output should be a simple short phrase describing the common overall intent of these utterances, "
                "replacing any proper name or specification with the category of the object (e.g., when given the name Alice, replace it with 'Customer Name'): \n"
            )
            for idx, utterance in enumerate(utterances):
                prompt += f"- {utterance}\n"
            prompt += "\nYour response should be a dict with one attribute that is 'intent'."
            completion = client.get_response(
                messages=[
                    {"role": "system", "content": "You are an AI agent specializing in intent and motive recognition, tasked with extracting intents from customer support conversations and outputting a JSON with one key 'intent'."},
                    {"role": "user", "content": prompt}
                ]
            )
            completion = pre_process_llm_output(completion)
            try:
                intent_by_cluster[str(cluster)] = completion["intent"]
            except Exception as e:
                print(f"Error reading completion response - completion response: {completion}")
                intent_by_cluster[str(cluster)] = completion

        return intent_by_cluster

    def add_intents_to_conversations(data: Dict[str, List[Dict[str, Any]]], 
                                     intents_by_cluster: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Adds intents to conversations based on cluster labels.

        Args:
            data (Dict[str, List[Dict[str, Any]]]): Data dictionary with `conv-id`s as keys and lists of dictionaries containing `utterance` and `cluster` attributes.
            intents_by_cluster (Dict[str, str]): Dictionary mapping cluster indices to intents.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Updated data dictionary with ordered intents added.
        """
        for key in data:
            intents_ordered = []
            for item in data[key]:
                if 'ordered_intents' not in item.keys() and len(item) != 0:
                    cluster = str(item['cluster'])
                    intent = intents_by_cluster.get(cluster, "Unknown")  # Handle missing intents
                    intents_ordered.append(intent)
            data[key].append({"ordered_intents": intents_ordered})
        return data

    def print_updated_data_with_ordered_intents(data: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Prints the updated data with ordered intents.

        Args:
            data (Dict[str, List[Dict[str, Any]]]): Data dictionary with `conv-id`s as keys and lists of dictionaries containing `utterance`, `cluster`, and `ordered_intents` attributes.
        """
        for key in data:
            print(f"Conversation: {key}")
            for item in data[key]:
                if "utterance" in item:
                    print(f"  Utterance: {item['utterance']}, Cluster: {item['cluster']}")
                else:
                    print(f"  Ordered Intents: {item['ordered_intents']}")

    def extract_ordered_intents(data: Dict[str, List[Dict[str, Any]]]) -> List[List[str]]:
        """
        Extracts ordered intents from the data.

        Args:
            data (Dict[str, List[Dict[str, Any]]]): Data dictionary with `conv-id`s as keys and lists of dictionaries containing `ordered_intents`.

        Returns:
            List[List[str]]: List of ordered intents extracted from the data.
        """
        ordered_intents = []
        for key in data:
            for item in data[key]:
                if "ordered_intents" in item:
                    ordered_intents.append(item["ordered_intents"])
        return ordered_intents

    
    def extract_verb_phrases(utterance: str) -> List[str]:
        """
        Extract verb phrases from a given utterance using spaCy.

        Args:
            utterance (str): The text of the utterance.

        Returns:
            List[str]: List of verb phrases in the utterance.
        """

        doc = nlp(utterance["content"])
        verb_phrases = []

        # Iterate over tokens and extract verb phrases using dependency parsing
        for token in doc:
            if 'VERB' in token.pos_ and token.dep_ in ['ROOT', 'xcomp', 'ccomp']:  # Identify main verbs and their complements
                verb_phrase = token.lemma_
                verb_phrases.append(verb_phrase)
        return verb_phrases
