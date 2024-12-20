import networkx as nx 
from typing import Dict, Any
import numpy as np
from tqdm import tqdm


class StructuralEvaluator:
    """
    A class for evaluating structural properties of a directed graph, including hyperbolicity,
    branching factor, and number of cycles.
    """

    @staticmethod
    def hyperbolicity_sample(G: nx.DiGraph, num_samples: int = 500) -> float:
        """
        Estimate the hyperbolicity of the given directed graph using a random sampling 
        of quadruples of nodes.

        Parameters
        ----------
        G : nx.DiGraph
            The input directed graph.
        num_samples : int, optional
            Number of random quadruples to sample (default: 500).

        Returns
        -------
        float
            The estimated hyperbolicity value. Zero if no valid quadruples could be sampled.
        """
        hyps = []
        nodes = list(G.nodes())
        
        if len(nodes) < 4:
            return 0.0

        for _ in tqdm(range(num_samples), desc="Sampling for hyperbolicity"):
            node_tuple = np.random.choice(nodes, 4, replace=False)
            s = []
            try:
                d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
                d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
                d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
                d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
                d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
                d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)

                s.append(d01 + d23)
                s.append(d02 + d13)
                s.append(d03 + d12)
                s.sort()
                hyps.append((s[-1] - s[-2]) / 2.0)
            except nx.NetworkXNoPath:
                # If no path exists between the chosen nodes, skip this quadruple
                continue

        return max(hyps) if hyps else 0.0

    @staticmethod
    def evaluate(G: nx.DiGraph, num_samples: int = 500) -> Dict[str, Any]:
        """
        Evaluate structural properties of the given directed graph: 
        delta hyperbolicity, branching factor, and the number of cycles.

        Parameters
        ----------
        G : nx.DiGraph
            The input directed graph.
        num_samples : int, optional
            Number of samples to estimate hyperbolicity (default: 5000).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            {
                "delta_hyperbolicity": float,
                "branching_factor": float,
                "num_cycles": int
            }
        """
        print("structural delta hyperbolicity calculation...")
        # Compute delta hyperbolicity
        delta_hyperbolicity = StructuralEvaluator.hyperbolicity_sample(G, num_samples=num_samples)
        print("branching factor and cycles count calculation...")
        # Compute branching factor: average out-degree = |E| / |V|
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        branching_factor = (num_edges / num_nodes) if num_nodes > 0 else 0.0

        # Compute number of cycles using simple_cycles
        cycles = list(nx.simple_cycles(G))
        num_cycles = len(cycles)

        return {
            "delta_hyperbolicity": delta_hyperbolicity,
            "branching_factor": branching_factor,
            "num_cycles": num_cycles
        }
