from src.graph.conversational_graph_builder import ConversationalGraphBuilder
import numpy as np
import networkx as nx
from typing import Dict


class AdaptiveThresholdGraphBuilder(ConversationalGraphBuilder):
    def create_directed_graph(
        transition_matrix: np.ndarray,
        intent_by_cluster: Dict[str, str],
        **kwargs,
    ) -> nx.DiGraph:
        """
        Creates a directed graph based on a transition matrix after applying
        alpha-based thresholding on row and column averages.

        Args:
            transition_matrix (np.ndarray): Matrix representing transition weights between clusters.
            intent_by_cluster (Dict[str, str]): Dictionary mapping cluster indices to their intents.
            kwargs (dict , optional): 
                - alpha (float): Scaling factor for thresholding. Cells with values 
                  < alpha * row or column average will be set to 0.

        Returns:
            nx.DiGraph: The created directed graph.
        """
        G = nx.DiGraph()
        
        try:
            alpha = kwargs["alpha"]
        except KeyError:
            raise ValueError("Alpha parameter is required for this method.")
        
        # Step 1: Calculate row and column averages
        row_averages = np.mean(transition_matrix, axis=1)
        column_averages = np.mean(transition_matrix, axis=0)
        
        # Step 2: Create a modified copy of the transition matrix
        modified_matrix = transition_matrix.copy()
        
        # Step 3: Apply alpha-based thresholding
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[1]):
                if (
                    transition_matrix[i, j] < alpha * row_averages[i] and
                    transition_matrix[i, j] < alpha * column_averages[j]
                ):
                    modified_matrix[i, j] = 0
        
        # Step 4: Build the graph using the modified matrix
        for i, from_intent in intent_by_cluster.items():
            for j, to_intent in intent_by_cluster.items():
                if modified_matrix[int(i), int(j)] > 0:
                    G.add_edge(from_intent, to_intent, weight=modified_matrix[int(i), int(j)])
        
        return G
