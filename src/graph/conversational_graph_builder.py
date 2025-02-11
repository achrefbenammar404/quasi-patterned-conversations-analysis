import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import os
from abc import ABC, abstractmethod
from matplotlib import cm, colors

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



    def plot_graph_html(G: nx.DiGraph, dir_name: str, file_name: str) -> None:
        """
        Generates an HTML visualization of the directed graph using PyVis and saves it to a file.
        Nodes are colored by community, and edges are colored based on their weights.

        Args:
            G (nx.DiGraph): The directed graph to be visualized.
            dir_name (str): Directory to save the output HTML file.
            file_name (str): Name of the output HTML file (without extension).
        """
        # Create PyVis network
        net = Network(notebook=True, width="100%", height="700px", directed=True, cdn_resources="in_line")
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        for node in G.nodes:
            net.add_node(node, label=str(node))

        # Find the minimum and maximum weights
        min_weight = float('inf')
        max_weight = float('-inf')
        for _, _, data in G.edges(data=True):
            weight = data.get('weight', 1)
            min_weight = min(min_weight, weight)
            max_weight = max(max_weight, weight)


        # Add edges with weight-based colors
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            net.add_edge(u, v, value=weight, title=f'weight: {weight:.2f}', color='blue')

        # Set options for better visualization
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
                        "scaleFactor": 0.5
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
                "enabled": false,
                "filter": "nodes,edges,physics",
                "showButton": true
            }
        }
        """)

        # Save the visualization as an HTML file
        file_path = os.path.join(dir_name, file_name + ".html")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(net.generate_html())
        


        print(f"Graph visualization saved to {file_path}")

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