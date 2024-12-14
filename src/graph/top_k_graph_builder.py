from src.graph.conversational_graph_builder import ConversationalGraphBuilder
import networkx as nx
import numpy as np
from typing import Dict


class ThresholdGraphBuilder(ConversationalGraphBuilder) : 
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
            kwargs (dict , optional): {tau :(float, optional): Minimum weight threshold for adding edges ,
            top_k (int, optional): Number of top edges to keep for each node if using 'top_k' algorithm}

        Returns:
            nx.DiGraph: The created directed graph.
        """
        G = nx.DiGraph()
        try : 
            tau = kwargs['tau']
            top_k = kwargs['top_k']
        except KeyError as ke : 
            print("Error occured while extracting params for top_k filtering : {ke}")

        for i, from_intent in intent_by_cluster.items():
                weights = transition_matrix[int(i)]
                top_indices = weights.argsort()[-top_k:][::-1]
                for j in top_indices:
                    if int(i) != int(j) and weights[j] > tau :
                        to_intent = intent_by_cluster[str(j)]
                        G.add_edge(from_intent, to_intent, weight=weights[j])
        return G 