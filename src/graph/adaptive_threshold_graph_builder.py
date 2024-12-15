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
        alpha-based thresholding on row and column statistics (mean and standard deviation).

        Args:
            transition_matrix (np.ndarray): Matrix representing transition weights between clusters.
            intent_by_cluster (Dict[str, str]): Dictionary mapping cluster indices to their intents.
            kwargs (dict, optional):
                - alpha (float): Scaling factor for thresholding. Cells with values
                  < alpha * std_deviation + mean (row/column) will be set to 0.

        Returns:
            nx.DiGraph: The created directed graph.
        """
        G = nx.DiGraph()

        try:
            alpha = kwargs["alpha"]
        except KeyError:
            raise ValueError("Alpha parameter is required for this method.")

        # Step 1: Calculate row and column statistics (mean and std deviation)
        row_means = np.mean(transition_matrix, axis=1)
        row_stds = np.std(transition_matrix, axis=1)

        column_means = np.mean(transition_matrix, axis=0)
        column_stds = np.std(transition_matrix, axis=0)

        # Step 2: Create a modified copy of the transition matrix
        modified_matrix = transition_matrix.copy()

        # Step 3: Apply alpha-based thresholding based on mean and std deviation
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[1]):
                row_threshold = row_means[i] + alpha * row_stds[i]
                column_threshold = column_means[j] + alpha * column_stds[j]

                if (
                    transition_matrix[i, j] < row_threshold or
                    transition_matrix[i, j] < column_threshold
                ):
                    modified_matrix[i, j] = 0

        # Step 4: Build the graph using the modified matrix
        for i, from_intent in intent_by_cluster.items():
            for j, to_intent in intent_by_cluster.items():
                if modified_matrix[int(i), int(j)] > 0:
                    G.add_edge(from_intent, to_intent, weight=modified_matrix[int(i), int(j)])

        return G
