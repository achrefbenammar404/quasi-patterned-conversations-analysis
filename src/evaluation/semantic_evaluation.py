import networkx as nx 
from typing import List , Dict , Any 
from src.utils.utils import read_json_to_dict , save_dict_to_json
from sentence_transformers import SentenceTransformer
import os 
import numpy as np 
from scipy import spatial

cache = read_json_to_dict(os.path.join("data" , "embedding_cache.json"))

class SemanticEvaluator:
    """
    A class for evaluating flow coverage and semantic coverage based on 
    provided flow graphs, reference flows, and their corresponding utterances and intents.
    """

    @staticmethod
    def evaluate(
        graph: nx.DiGraph,
        ordered_intents: List[List[str]],
        ordered_utterances: List[List[str]],
        model: SentenceTransformer
    ) -> Dict[str, float]:
        """
        Evaluate both structural coverage and semantic coverage for given flows.

        Parameters
        ----------
        graph : nx.DiGraph
            The directed graph representing the flow structure.
        ordered_intents : List[List[str]]
            A list of flows, each flow is a list of intent strings, ordered as they appear in the conversation.
        ordered_utterances : List[List[str]]
            A list of flows, each flow is a list of utterance strings, aligned with the given intents.
        real_flows : List[List[str]]
            A list of flows from real data (reference flows), each flow is a list of node identifiers.
        model : SentenceTransformer
            A sentence transformer model used to encode utterances and intents for semantic similarity calculation.

        Returns
        -------
        Dict[str, float]
            A dictionary containing:
            {
                "coverage": float,           # The ratio of transitions in the real flows that are present in the graph
                "semantic_coverage": float   # The average cosine similarity-based semantic coverage between intents and utterances
            }
        """

        coverage_score = SemanticEvaluator.calculate_coverage(
            graph, 
            ordered_intents
        )
        semantic_coverage_score = SemanticEvaluator.calculate_semantic_coverage(
            ordered_intents, 
            ordered_utterances, 
            model
        )
        return {
            "coverage": coverage_score,
            "semantic_coverage": semantic_coverage_score
        }

    @staticmethod
    def calculate_semantic_coverage(
        ordered_intents: List[List[str]],
        ordered_utterances: List[List[str]],
        model: SentenceTransformer
    ) -> float:
        """
        Calculate semantic coverage by computing the average cosine similarity between 
        intent embeddings and utterance embeddings across all flows.

        Parameters
        ----------
        ordered_intents : List[List[str]]
            A list of flows, each flow is a list of intent strings.
        ordered_utterances : List[List[str]]
            A list of flows, each flow is a list of utterance strings, aligned with the given intents.
        model : SentenceTransformer
            A sentence transformer model used to encode utterances and intents.

        Returns
        -------
        float
            The average cosine similarity score (ranging from -1 to 1) over all (intent, utterance) pairs.
        """
        score = 0.0
        num = 0
        for intents, utterances in zip(ordered_intents, ordered_utterances):
            for intent, utterance in zip(intents, utterances):
                if intent in cache:
                    emb_intent = np.array(cache[intent])
                else:
                    emb_intent = model.encode(sentences=[intent])
                    cache[intent] = emb_intent.tolist()

                if utterance in cache:
                    emb_utterance = np.array(cache[utterance])
                else:
                    emb_utterance = model.encode(sentences=[utterance])
                    cache[utterance] = emb_utterance.tolist()
                similarity = 1 - spatial.distance.cosine(emb_utterance[0], emb_intent[0])
                score += similarity
                num += 1
        save_dict_to_json(cache , os.path.join("data" , "embedding_cache.json") )
        if num > 0:
            return score / num
        return 0.0

    @staticmethod
    def calculate_coverage(
        graph: nx.DiGraph, 
        real_flows: List[List[str]]
    ) -> float:
        """
        Calculate the structural coverage of given real flows within the provided graph.

        The coverage is defined as the average automation rate (ratio of consecutive 
        pairs of nodes in the real flow that appear as edges in the graph) across all real flows.

        Parameters
        ----------
        graph : nx.DiGraph
            The directed graph representing the flow structure.
        real_flows : List[List[str]]
            A list of flows, each flow is a list of node identifiers in the order they occur in the conversation.

        Returns
        -------
        float
            The coverage metric: average automation rate across all real flows.
        """
        matched_flows = 0.0
        for flow in real_flows:
            matched_flows += SemanticEvaluator.automation_rate(graph, flow)

        if len(real_flows) > 0:
            coverage = matched_flows / len(real_flows)
        else:
            coverage = 0.0
        return coverage

    @staticmethod
    def automation_rate(
        graph: nx.DiGraph, 
        flow: List[str]
    ) -> float:
        """
        Compute the automation rate for a single flow.

        The automation rate is the ratio of transitions in the given flow that are 
        represented as edges in the provided graph.

        Parameters
        ----------
        graph : nx.DiGraph
            The directed graph representing the flow structure.
        flow : List[str]
            A single flow sequence, each element is a node identifier.

        Returns
        -------
        float
            The automation rate for the provided flow.
        """
        matched_transitions = 0
        total_transitions = len(flow) - 1

        for i in range(total_transitions):
            if graph.has_edge(flow[i], flow[i + 1]):
                matched_transitions += 1

        if total_transitions > 0:
            return matched_transitions / total_transitions
        return 0.0
