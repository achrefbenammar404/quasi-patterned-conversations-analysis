import networkx as nx
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from src.evaluation.semantic_evaluation import SemanticEvaluator
from src.evaluation.structural_evaluation import StructuralEvaluator

class Evaluator:
    """
    A unified evaluation class for assessing both semantic and structural properties
    of a directed graph representing conversational flows or similar structures.
    """

    @staticmethod
    def evaluate(
        graph: nx.DiGraph,
        ordered_intents: List[List[str]],
        ordered_utterances: List[List[str]],
        model: SentenceTransformer,
        num_samples: int = 5000
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the semantic and structural properties of a directed graph.

        This method combines two evaluation dimensions:
        1. **Semantic Evaluation**: Measures structural coverage and semantic alignment 
           between intents and utterances using the provided SentenceTransformer model.
        2. **Structural Evaluation**: Computes graph properties like hyperbolicity, branching factor, 
           and number of cycles.

        Parameters
        ----------
        graph : nx.DiGraph
            The directed graph representing the flow structure.
        ordered_intents : List[List[str]]
            A list of flows, where each flow is a list of intent strings, ordered as they appear in the conversation.
        ordered_utterances : List[List[str]]
            A list of flows, where each flow is a list of utterance strings aligned with the given intents.
        model : SentenceTransformer
            A pre-trained sentence transformer model used to compute semantic similarity.
        num_samples : int, optional
            Number of samples to estimate hyperbolicity for the structural evaluation (default: 5000).

        Returns
        -------
        Dict[str, Dict[str, float]]
            A dictionary containing:
            {
                "semantic_scores": {
                    "coverage": float,           # Ratio of real-flow transitions covered by the graph
                    "semantic_coverage": float   # Average cosine similarity between intents and utterances
                },
                "structural_scores": {
                    "delta_hyperbolicity": float,  # Estimated hyperbolicity of the graph
                    "branching_factor": float,     # Average out-degree of nodes in the graph
                    "num_cycles": int             # Number of cycles in the graph
                }
            }

        Examples
        --------
        >>> graph = nx.DiGraph()
        >>> graph.add_edges_from([("start", "intent1"), ("intent1", "intent2"), ("intent2", "end")])
        >>> ordered_intents = [["start", "intent1", "intent2", "end"]]
        >>> ordered_utterances = [["Hi", "I need help", "Sure", "Goodbye"]]
        >>> model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        >>> Evaluator.evaluate(graph, ordered_intents, ordered_utterances, model)
        {
            "semantic_scores": {"coverage": 1.0, "semantic_coverage": 0.87},
            "structural_scores": {
                "delta_hyperbolicity": 0.5,
                "branching_factor": 1.0,
                "num_cycles": 0
            }
        }
        """
        # Evaluate semantic scores
        semantic_scores = SemanticEvaluator.evaluate(
            graph=graph,
            ordered_intents=ordered_intents,
            ordered_utterances=ordered_utterances,
            model=model
        )

        # Evaluate structural scores
        structural_scores = StructuralEvaluator.evaluate(
            G=graph,
            num_samples=num_samples
        )

        return {
            "semantic_scores": semantic_scores,
            "structural_scores": structural_scores
        }
