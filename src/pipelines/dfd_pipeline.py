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
    AdaptiveThresholdGraphBuilder,
    FilterReconnectGraphBuilder,
    ThresholdGraphBuilder,
    TopKGraphBuilder
)
from src.evaluation.evaluator import Evaluator

graph_builders: Dict[str, ConversationalGraphBuilder] = {
    'adaptive_threshold_graph_builder': AdaptiveThresholdGraphBuilder,
    'filter_reconnect_graph_builder': FilterReconnectGraphBuilder,
    'threshold_graph_builder': ThresholdGraphBuilder,
    'top_k_graph_builder': TopKGraphBuilder
}

def run(
    dataset_name: str,
    train_data,
    test_data,
    min_clusters,
    max_clusters,
    model: SentenceTransformer,
    tau,
    n_closest,
    label_model,
):
    print(f"Starting the pipeline for dataset: {dataset_name}")

    # Embed the sampled data
    print("Embedding train data...")
    data = ExtractEmbed.embed_sampled_data(train_data, model, dataset_name=dataset_name)
    print(f"Embedding complete. Total conversations embedded: { len(data)}")

    # Extract embeddings
    print("Extracting embeddings from train data...")
    all_embeddings = ExtractEmbed.extract_embeddings(data)
    print("Embeddings extracted")

    # Determine optimal number of clusters
    if min_clusters != max_clusters:
        print(f"Using the elbow method to determine optimal clusters (min: {min_clusters}, max: { max_clusters})")
        Cluster.elbow_method(all_embeddings, min_clusters=min_clusters, max_clusters=max_clusters)
        optimal_cluster_number = int(input("Optimal cluster number: "))
    else:
        optimal_cluster_number = min_clusters
    print(f"Optimal cluster number selected: { optimal_cluster_number}")

    # Cluster the embeddings
    print("Clustering embeddings...")
    clustered_data, embeddings, labels, cluster_centers = Cluster.cluster_embeddings(
        data, num_clusters=optimal_cluster_number
    )
    print(f"Clustering complete. Number of clusters: {len(cluster_centers)}" )

    # Extract closest utterances
    print("Extracting closest utterances for each cluster...")
    closest_utterances = Cluster.extract_closest_embeddings(
        clustered_data, embeddings, labels, cluster_centers, n=n_closest
    )
    print("Closest utterances extracted.")

    # Label clusters
    print("Labeling clusters using the specified model...")
    intent_by_cluster = Label.label_clusters_by_closest_utterances(closest_utterances, model=label_model)
    print(f"Cluster labeling complete. Total intents: {len(intent_by_cluster)}" )

    # Add intents to conversations
    print("Adding intents to conversations...")
    updated_data_with_intents = Label.add_intents_to_conversations(clustered_data, intent_by_cluster)
    print("Intents added to conversations.")

    # Extract ordered intents
    print("Extracting ordered intents...")
    ordered_intents = Label.extract_ordered_intents(updated_data_with_intents)
    print(f"Ordered intents extracted. Total: {len(ordered_intents)}" )

    # Create transition matrix
    print("Creating transition matrix...")
    transition_matrix = TransitionAnalysis.create_transition_matrix(ordered_intents, intent_by_cluster)
    print("Transition matrix created.")

    builder = graph_builders["threshold_graph_builder"]
    # Build graph
    print(f"Creating directed graph with tau={tau}" )
    graph = builder.create_directed_graph(
        transition_matrix=transition_matrix,
        intent_by_cluster=intent_by_cluster,
        tau=0
    )
    print("Directed graph created.")

    # Handle test data
    print("Embedding test data...")
    test_utterances = ExtractEmbed.extract_utterances(test_data)
    test_data = ExtractEmbed.embed_sampled_data(test_data, model, dataset_name=dataset_name)
    test_all_embeddings = ExtractEmbed.extract_embeddings(test_data)
    print("Test data embedded")

    print("Assigning test data to clusters...")
    test_data_assigned_cluster_ids = Cluster.assign_to_clusters(cluster_centers, test_all_embeddings)
    print("Test data assigned to clusters.")

    # Evaluate the graph
    print("Evaluating the graph...")
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
        num_samples=5000
    )
    print(f"Evaluation complete. Scores: { str(scores)}")
    return graph, scores
