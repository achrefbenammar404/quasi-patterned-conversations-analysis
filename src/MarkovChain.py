import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import numpy as np
from typing import Dict, List, Tuple
import os

class ConversationalGraph:

    def plot_graph_html(G: nx.DiGraph, file_name: str) -> None:
        """
        Generates an HTML visualization of the directed graph using PyVis and saves it to a file.

        Args:
            G (nx.DiGraph): The directed graph to be visualized.
            file_name (str): Name of the output HTML file (without extension).
        """
        net = Network(notebook=True, width="100%", height="700px", directed=True, cdn_resources="in_line")

        # Add nodes and edges from NetworkX graph to PyVis network
        for node in G.nodes:
            net.add_node(node, label=str(node), title=str(node))

        # Find the minimum and maximum weights
        min_weight = float('inf')
        max_weight = float('-inf')
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            min_weight = min(min_weight, weight)
            max_weight = max(max_weight, weight)

        # Normalize the weight to a 0-1 scale and map to color
        def get_edge_color(weight: float) -> str:
            normalized_weight = (weight - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0
            color = f'rgb({int(255 * normalized_weight)}, 0, {int(255 * (1 - normalized_weight))})'
            return color

        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            color = get_edge_color(weight)
            net.add_edge(u, v, value=weight, title=f'weight: {weight:.2f}', color=color)

        # Set options for better visualization and enable the physics control panel
        net.set_options("""
        var options = {
            "nodes": {
                "font": {
                    "size": 20
                }
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 1
                    }
                },
                "font": {
                    "size": 14,
                    "align": "horizontal"
                },
                "smooth": {
                    "enabled": true,
                    "type": "dynamic"
                }
            },
            "physics": {
                "enabled": true,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "theta": 0.5,
                    "gravitationalConstant": -86,
                    "centralGravity": 0.005,
                    "springLength": 120,
                    "springConstant": 0.04,
                    "damping": 0.57,
                    "avoidOverlap": 0.92
                },
                "maxVelocity": 41,
                "minVelocity": 1,
                "timestep": 0.5,
                "wind": {
                    "x": 0,
                    "y": 0
                }
            },
            "configure": {
                "enabled": true,
                "filter": "nodes,edges,physics",
                "showButton": true
            }
        }
        """)

        # Generate the graph and save it to an HTML file
        if not os.path.exists("output"):
            os.mkdir("output")
        net.show(f"output/{file_name}.html")

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

        G = ConversationalGraph.remove_weakest_edge_in_cycles(G)
        G = ConversationalGraph.reconnect_subgraphs(G, transition_matrix, intent_by_cluster)

        return G

    def create_directed_graph(
        transition_matrix: np.ndarray, 
        intent_by_cluster: Dict[str, str], 
        min_weight: float = 0.1,
        algorithm: str = 'threshold',
        top_k: int = 5
    ) -> nx.DiGraph:
        """
        Creates a directed graph based on a transition matrix, intent mapping, and a specified algorithm.

        Args:
            transition_matrix (np.ndarray): Matrix representing transition weights between clusters.
            intent_by_cluster (Dict[str, str]): Dictionary mapping cluster indices to their intents.
            min_weight (float, optional): Minimum weight threshold for adding edges. Defaults to 0.1.
            algorithm (str, optional): Algorithm to use for adding edges ('threshold', 'top_k', 'filter&reconnect'). Defaults to 'threshold'.
            top_k (int, optional): Number of top edges to keep for each node if using 'top_k' algorithm. Defaults to 5.

        Returns:
            nx.DiGraph: The created directed graph.
        """
        G = nx.DiGraph()

        if algorithm == 'threshold':
            # Add edges with weights greater than the specified minimum weight
            for i, from_intent in intent_by_cluster.items():
                for j, to_intent in intent_by_cluster.items():
                    if int(i) != int(j) and transition_matrix[int(i), int(j)] > min_weight:
                        G.add_edge(from_intent, to_intent, weight=transition_matrix[int(i), int(j)])
        elif algorithm == 'top_k':
            # Add top-K edges for each node
            for i, from_intent in intent_by_cluster.items():
                weights = transition_matrix[int(i)]
                top_indices = weights.argsort()[-top_k:][::-1]
                for j in top_indices:
                    if int(i) != int(j) and weights[j] > min_weight:
                        to_intent = intent_by_cluster[str(j)]
                        G.add_edge(from_intent, to_intent, weight=weights[j])
        elif algorithm == "filter&reconnect":
            G = ConversationalGraph.filter_and_reconnect(transition_matrix, intent_by_cluster, min_weight, top_k)

        return G
