import argparse
import logging
from src.clustering import Cluster
from src.embedding import ExtractEmbed
from src.label import Label
from src.transition_matrix import TransitionAnalysis
from src.utils.utils import * 
from sentence_transformers import SentenceTransformer
from src.graph import (
    ConversationalGraphBuilder,
    ThresholdGraphBuilder
)
from src.evaluation.evaluator import Evaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run(
    dataset_name: str,
    train_data,
    test_data,
    min_clusters,
    max_clusters,
    model: SentenceTransformer,
    tau
):
    builder = ThresholdGraphBuilder
    logging.info("Starting the pipeline for dataset: %s", dataset_name)

    # Embed the sampled data
    logging.info("Embedding train data...")
    data = ExtractEmbed.embed_sampled_data(train_data, model, dataset_name=dataset_name)
    logging.info("Embedding complete. Total conversations embedded: %d", len(data))

    # Extract embeddings
    logging.info("Extracting embeddings...")
    all_embeddings = ExtractEmbed.extract_embeddings(data)
    logging.info("Embeddings extracted. Shape: %s", str(all_embeddings.shape))

    # Determine the optimal number of clusters
    if min_clusters != max_clusters:
        logging.info("Using the elbow method to determine optimal clusters (min: %d, max: %d)", min_clusters, max_clusters)
        Cluster.elbow_method(all_embeddings, min_clusters=min_clusters, max_clusters=max_clusters)
        optimal_cluster_number = int(input("Optimal cluster number: "))
    else:
        optimal_cluster_number = min_clusters
    logging.info("Optimal cluster number selected: %d", optimal_cluster_number)

    # Cluster the embeddings
    logging.info("Clustering embeddings...")
    clustered_data, embeddings, labels, cluster_centers = Cluster.cluster_embeddings(
        data, num_clusters=optimal_cluster_number
    )
    logging.info("Clustering complete. Number of clusters: %d", len(cluster_centers))

    # Extract closest utterances
    logging.info("Extracting closest utterances for each cluster...")
    closest_utterances = Cluster.extract_closest_embeddings(
        clustered_data, embeddings, labels, cluster_centers, n=10**100
    )
    logging.info("Closest utterances extracted.")

    # Label clusters
    logging.info("Labeling clusters...")
    intent_by_cluster = Label.label_clusters_by_verbphrases(closest_utterances)
    logging.info("Cluster labeling complete. Total intents: %d", len(intent_by_cluster))

    # Add intents to conversations
    logging.info("Adding intents to conversations...")
    updated_data_with_intents = Label.add_intents_to_conversations(clustered_data, intent_by_cluster)
    logging.info("Intents added to conversations.")

    # Extract ordered intents
    logging.info("Extracting ordered intents...")
    ordered_intents = Label.extract_ordered_intents(updated_data_with_intents)
    logging.info("Ordered intents extracted. Total: %d", len(ordered_intents))

    # Create transition matrix
    logging.info("Creating transition matrix...")
    transition_matrix = TransitionAnalysis.create_transition_matrix(ordered_intents, intent_by_cluster)
    logging.info("Transition matrix created.")

    # Create and plot conversational graph
    logging.info("Creating directed graph with tau=%f...", tau)
    graph = builder.create_directed_graph(
        transition_matrix=transition_matrix,
        intent_by_cluster=intent_by_cluster,
        tau=tau,
    )
    logging.info("Directed graph created.")

    # Handle test data
    logging.info("Embedding test data...")
    test_utterances = ExtractEmbed.extract_utterances(test_data)
    test_data = ExtractEmbed.embed_sampled_data(test_data, model, dataset_name=dataset_name)
    test_all_embeddings = ExtractEmbed.extract_embeddings(test_data)
    logging.info("Test data embedded. Shape: %s", str(test_all_embeddings.shape))

    logging.info("Assigning test data to clusters...")
    test_data_assigned_cluster_ids = Cluster.assign_to_clusters(cluster_centers, test_all_embeddings)
    logging.info("Test data assigned to clusters.")

    # Evaluate the model
    logging.info("Evaluating the model...")
    test_ordered_intents = []
    counter = 0
    for conv in test_utterances:
        intents = []
        for utterance in conv:
            intents.append(intent_by_cluster[str(test_data_assigned_cluster_ids[counter])])
            counter += 1
        test_ordered_intents.append(intents)
    scores = Evaluator.evaluate(
        graph,
        ordered_intents=test_ordered_intents,
        ordered_utterances=test_utterances,
        model=model,
        num_samples=5000,
    )
    logging.info("Evaluation complete. Scores: %s", str(scores))

    return graph, scores
