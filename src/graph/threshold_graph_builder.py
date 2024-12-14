from src.graph.conversational_graph_builder import ConversationalGraph
import networkx as nx
import numpy as np
from typing import Dict


class ThresholdGraphBuilder(ConversationalGraph) : 
    def create_directed_graph(
        transition_matrix: np.ndarray, 
        intent_by_cluster: Dict[str, str], 
        **kwargs 
    ) -> nx.DiGraph:
        """
        Creates a directed graph based on a transition matrix, intent mapping, and a specified algorithm.

        Args:
            transition_matrix (np.ndarray): Matrix representing transition weights between clusters.
            intent_by_cluster (Dict[str, str]): Dictionary mapping cluster indices to their intents.
            kwargs (dict , optional): {tau :(float, optional): Minimum weight threshold for adding edges. }

        Returns:
            nx.DiGraph: The created directed graph.
        """
        G = nx.DiGraph()
        try : 
            tau = kwargs['tau']
        except KeyError as ke : 
            print("Error occured while extracting params for threshold filtering : {ke}")
        for i, from_intent in intent_by_cluster.items():
            for j, to_intent in intent_by_cluster.items():
                if transition_matrix[int(i), int(j)] > tau:
                    G.add_edge(from_intent, to_intent, weight=transition_matrix[int(i), int(j)])
        return G 