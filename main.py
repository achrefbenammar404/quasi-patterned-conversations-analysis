import argparse
from src.clustering import Cluster
from src.embedding import ExtractEmbed
from src.label import Label
from src.transition_matrix import TransitionAnalysis
from src.utils.utils import * 
from sentence_transformers import SentenceTransformer
from src.graph import (
    ConversationalGraphBuilder , 
    AdaptiveThresholdGraphBuilder , 
    FilterReconnectGraphBuilder , 
    ThresholdGraphBuilder , 
    TopKGraphBuilder
)


graph_builders  : Dict[str , ConversationalGraphBuilder ]= {
    'adaptive_threshold_graph_builder' : AdaptiveThresholdGraphBuilder , 
    'filter_reconnect_graph_builder' : FilterReconnectGraphBuilder , 
    'threshold_graph_builder' : ThresholdGraphBuilder , 
    'top_k_graph_builder' : TopKGraphBuilder
}


def main(args):
    # Read the conversations from the JSON file

    conversations = read_json_file(args.file_path)

    # Extract customer support utterances
    customer_support_utterances = ExtractEmbed.extract_customer_support_utterances(conversations)
    
    # Sample data points
    sampled_data = get_random_n_pairs(customer_support_utterances, n=args.num_sampled_data)
    
    # Load the sentence transformer model
    model = SentenceTransformer(model_name_or_path=args.model_name)
    
    # Embed the sampled data
    data = ExtractEmbed.embed_sampled_data(sampled_data, model)
    
    # Extract embeddings
    all_embeddings = ExtractEmbed.extract_embeddings(data)
    
    if args.min_clusters !=  args.max_clusters : 
        # Use the elbow method to determine optimal number of clusters
        Cluster.elbow_method(all_embeddings, min_clusters = args.min_clusters , max_clusters=args.max_clusters)
        
    # Ask for the optimal cluster number
        optimal_cluster_number = int(input("Optimal cluster number: "))
    else : 
        optimal_cluster_number = args.min_clusters 
    
    # Cluster the embeddings
    clustered_data, embeddings, labels, cluster_centers = Cluster.cluster_embeddings(data, num_clusters=optimal_cluster_number)
    
    # Visualize clusters before outlier removal
    Cluster.visualize_clusters_tsne(embeddings, labels, perplexity=50, title="t-SNE Visualization of Clusters Before Outlier Removal")
    
    # Plot distance distribution before outlier removal
    stats = Cluster.plot_distance_distribution(embeddings, labels, cluster_centers, title="Histogram of Distances to Cluster Centroids Before Outlier Removal")
    print(stats)

    
    # Visualize clusters after outlier removal
    Cluster.visualize_clusters_tsne(embeddings, labels, perplexity=50, title="t-SNE Visualization of Clusters After Outlier Removal")
    
    # Plot distance distribution after outlier removal
    Cluster.plot_distance_distribution(embeddings, labels, cluster_centers, title="Histogram of Distances to Cluster Centroids After Outlier Removal")
    
    # Extract closest utterances
    n_closest = args.n_closest
    closest_utterances = Cluster.extract_closest_embeddings(clustered_data, embeddings, labels, cluster_centers, n=n_closest)
    Cluster.print_closest_utterances(closest_utterances)
    
    # Label clusters
    intent_by_cluster = Label.label_clusters_by_closest_utterances(closest_utterances, model=args.label_model)
    
    # Generate cluster by intent
    cluster_by_intent = Label.generate_cluster_by_intent(intent_by_cluster)
    
    # Add intents to conversations
    updated_data_with_intents = Label.add_intents_to_conversations(clustered_data, intent_by_cluster)
    
    # Print updated data with ordered intents
    Label.print_updated_data_with_ordered_intents(updated_data_with_intents)
    
    # Extract ordered intents
    ordered_intents = Label.extract_ordered_intents(updated_data_with_intents)
    sampled_data = None 
    # Create transition matrix
    transition_matrix = TransitionAnalysis.create_transition_matrix(ordered_intents, intent_by_cluster)
    
    # Plot transition matrix
    #TransitionAnalysis.plot_transition_matrix(transition_matrix, intent_by_cluster)
    
    # Create and plot conversational graphs
    for builder_name , builder in graph_builders.items():
        graph = builder.create_directed_graph(
            transition_matrix = transition_matrix, 
            intent_by_cluster = intent_by_cluster, 
            tau=args.tau, 
            top_k=args.top_k, 
            alpha = args.alpha 
        )
        builder.plot_graph_html(graph, builder_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quasi-patterned Conversations Analysis")
    parser.add_argument("--file_path" , type = str , default = "data/ABCD.json" , help="path for json formatted conversations/dialogues" )
    parser.add_argument("--num_sampled_data", type=int, default=1000, help="Number of sampled datapoints")
    parser.add_argument("--max_clusters", type=int, default=15, help="Maximum number of clusters for the elbow method")
    parser.add_argument("--min_clusters" ,type=int, default=9, help="Minimum number of clusters for the elbow method" )
    parser.add_argument("--percentile", type=int, default=75, help="Percentile for outlier removal")
    parser.add_argument("--model_name", type=str, default='sentence-transformers/all-MiniLM-L12-v2', help="Model name for SentenceTransformer")
    parser.add_argument("--label_model", type=str, default='open-mixtral-8x22b', help="Model for labeling clusters by closest utterance")
    parser.add_argument("--tau", type=float, default=0, help="Minimum weight for conversational graph edges")
    parser.add_argument("--top_k", type=int, default=1, help="Top k edges to keep in the conversational graph")
    parser.add_argument("--alpha", type=int, default=1, help="alpha for adaptive threshold graph builder")
    parser.add_argument("--n_closest" , type=int , default=10, help="Number of the closest utterances to each cluster centroid to be passed to the llm for intent extraction")
    args = parser.parse_args()
    main(args)
