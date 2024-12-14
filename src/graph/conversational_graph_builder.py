import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import os
from abc import ABC, abstractmethod

class ConversationalGraphBuilder(ABC):
    @abstractmethod
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
            min_weight (float, optional): Minimum weight threshold for adding edges. Defaults to 0.1.
            algorithm (str, optional): Algorithm to use for adding edges ('threshold', 'top_k', 'filter&reconnect'). Defaults to 'threshold'.
            top_k (int, optional): Number of top edges to keep for each node if using 'top_k' algorithm. Defaults to 5.

        Returns:
            nx.DiGraph: The created directed graph.
        """
        pass 


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


    
    def create_sankey_diagram(g: nx.DiGraph, file_name):
        nodes = list(g.nodes)

        node_map = {node: i for i, node in enumerate(nodes)}

        source = []
        target = []
        value = []

        for u, v, data in g.edges(data=True):
            source.append(node_map[u])
            target.append(node_map[v])
            value.append(data['weight'])  

        # Create Sankey Diagram
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=30,  # Increased padding between nodes
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,  # Node labels
            ),
            link=dict(
                source=source,  # Source indices
                target=target,  # Target indices
                value=value,    
            )
        ))

        fig.update_layout(title_text="Sankey Diagram of Directed Graph", font_size=12)
        fig.write_html(f"{file_name}.html")