from src.graph.conversational_graph_builder import ConversationalGraphBuilder
import networkx as nx
import numpy as np
from typing import Dict


class TopKGraphBuilder(ConversationalGraphBuilder) : 
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
            G.add_node(from_intent)


        for intent in intent_by_cluster.values():
            G.add_node(intent)

        for i, from_intent in intent_by_cluster.items():
            weights =transition_matrix[int(i)]
            for j, weight in enumerate(weights):
                if weight < tau:
                    continue
                to_intent = intent_by_cluster[str(j)]
                G .add_edge(from_intent, to_intent, weight=weight)
        filtered_graph = G.copy()

        for node in G.nodes:
            outgoing_edges = [(node, neighbor, data) for neighbor, data in G[node].items()]
            outgoing_edges = sorted(outgoing_edges, key=lambda x: x[2].get('weight', 0), reverse=True)
            edges_to_keep = outgoing_edges[:top_k]
            neighbors_to_keep = {edge[1] for edge in edges_to_keep}
            for neighbor in list(G[node].keys()):
                if neighbor not in neighbors_to_keep:
                    filtered_graph.remove_edge(node, neighbor)

        return filtered_graph
