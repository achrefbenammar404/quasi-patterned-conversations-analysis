from src.graph.conversational_graph_builder import ConversationalGraphBuilder
import networkx as nx
import numpy as np
from typing import Dict


class FilterReconnectGraphBuilder(ConversationalGraphBuilder) : 
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
            print(f"Error occured while extracting params for filter and reconnect : {ke}")

        for i, from_intent in intent_by_cluster.items():
                weights = transition_matrix[int(i)]
                top_indices = weights.argsort()[-top_k:][::-1]
                for j in top_indices:
                    if int(i) != int(j) and weights[j] > tau :
                        to_intent = intent_by_cluster[str(j)]
                        G.add_edge(from_intent, to_intent, weight=weights[j])
        return G 
    def remove_weakest_edge_in_cycles(G: nx.DiGraph) -> nx.DiGraph:
        """
        Detects cycles in the directed graph and removes the weakest edge (edge with the lowest weight) in each cycle.

        Args:
            G (nx.DiGraph): The directed graph from which to remove weak edges in cycles.

        Returns:
            nx.DiGraph: The graph with the weakest edges in cycles removed.
        """
        try:
            cycles = list(nx.simple_cycles(G))
        except nx.NetworkXNoCycle:
            cycles = []

        while cycles:
            for cycle in cycles:
                edges_in_cycle = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
                valid_edges = [edge for edge in edges_in_cycle if G.has_edge(*edge)]

                if valid_edges:
                    weakest_edge = min(valid_edges, key=lambda edge: G.edges[edge]['weight'])
                    G.remove_edge(*weakest_edge)
            try:
                cycles = list(nx.simple_cycles(G))
            except nx.NetworkXNoCycle:
                cycles = []

        return G

    def reconnect_subgraphs(G: nx.DiGraph, transition_matrix: np.ndarray, intent_by_cluster: Dict[str, str]) -> nx.DiGraph:
        """
        Reconnects small detached subgraphs to the main graph using the transition matrix.

        Args:
            G (nx.DiGraph): The directed graph to be reconnected.
            transition_matrix (np.ndarray): Matrix representing transition weights between clusters.
            intent_by_cluster (Dict[str, str]): Dictionary mapping cluster indices to their intents.

        Returns:
            nx.DiGraph: The graph with subgraphs reconnected.
        """
        subgraphs = list(nx.weakly_connected_components(G))

        if len(subgraphs) <= 1:
            return G

        main_subgraph = max(subgraphs, key=len)

        for subgraph in subgraphs:
            if subgraph == main_subgraph:
                continue

            max_weight = -np.inf
            best_edge = None

            for node in subgraph:
                for main_node in main_subgraph:
                    node_idx = list(intent_by_cluster.keys())[list(intent_by_cluster.values()).index(node)]
                    main_node_idx = list(intent_by_cluster.keys())[list(intent_by_cluster.values()).index(main_node)]

                    weight = transition_matrix[int(node_idx), int(main_node_idx)]

                    if weight > max_weight:
                        max_weight = weight
                        best_edge = (node, main_node)

            if best_edge:
                G.add_edge(*best_edge, weight=max_weight)

        return G

    def filter_and_reconnect(
        transition_matrix: np.ndarray, 
        intent_by_cluster: Dict[str, str], 
        min_weight: float = 0,
        top_k: int = 5
    ) -> nx.DiGraph:
        """
        Filters edges based on weight and reconnects subgraphs using the transition matrix.

        Args:
            transition_matrix (np.ndarray): Matrix representing transition weights between clusters.
            intent_by_cluster (Dict[str, str]): Dictionary mapping cluster indices to their intents.
            min_weight (float, optional): Minimum weight threshold for keeping edges. Defaults to 0.
            top_k (int, optional): Number of top edges to keep for each node. Defaults to 5.

        Returns:
            nx.DiGraph: The directed graph after filtering and reconnecting subgraphs.
        """
        G = nx.DiGraph()

        # Add edges to the graph with weights above the minimum weight
        for i, from_intent in intent_by_cluster.items():
            weights = transition_matrix[int(i)]
            for j, weight in enumerate(weights):
                if weight >= min_weight and int(i) != int(j):
                    to_intent = intent_by_cluster[str(j)]
                    G.add_edge(from_intent, to_intent, weight=weight)

        # Keep only the top-k edges based on weight for each node
        incoming_edges = {}
        for u, v, data in G.edges(data=True):
            if v not in incoming_edges:
                incoming_edges[v] = []
            incoming_edges[v].append((u, data['weight']))

        for v, edges in incoming_edges.items():
            edges.sort(key=lambda x: x[1], reverse=True)
            top_edges = edges[:top_k]
            G.remove_edges_from([(u, v) for u, _ in edges])
            for u, weight in top_edges:
                G.add_edge(u, v, weight=weight)

        G = FilterReconnectGraphBuilder.remove_weakest_edge_in_cycles(G)
        G = FilterReconnectGraphBuilder.reconnect_subgraphs(G, transition_matrix, intent_by_cluster)

        return G

