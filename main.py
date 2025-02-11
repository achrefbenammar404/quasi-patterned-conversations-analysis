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
from src.evaluation.evaluator import Evaluator
import os 

graph_builders  : Dict[str , ConversationalGraphBuilder ]= {
    'adaptive_threshold' : AdaptiveThresholdGraphBuilder , 
    'filter_reconnect' : FilterReconnectGraphBuilder , 
    'threshold' : ThresholdGraphBuilder , 
    'top_k' : TopKGraphBuilder
}


def main(args):
    # Read the conversations from the JSON file

    conversations = read_json_file(args.file_path)

    # Extract customer support utterances
    customer_support_utterances = ExtractEmbed.extract_customer_support_utterances(conversations)
    
    # Sample data points
    sampled_data = get_random_n_pairs(customer_support_utterances, n=args.num_sampled_data)
    train_data, test_data = split_data(0.8, data=sampled_data)
    # Load the sentence transformer model
    model = SentenceTransformer(model_name_or_path=args.model_name , device ="cuda" )
    
    # Embed the sampled data
    data = ExtractEmbed.embed_sampled_data(train_data, model , dataset_name = args.file_path)
    
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
    
    # Extract closest utterances
    n_closest = args.n_closest
    closest_utterances = Cluster.extract_closest_embeddings(clustered_data, embeddings, labels, cluster_centers, n=n_closest)
    
    if args.approach in ["our_approach" , "ferreira2024"] :
    # Label clusters
        intent_by_cluster = Label.label_clusters_by_closest_utterances(closest_utterances, model=args.label_model)
    elif args.approach in ["carvalho2024"] : 
        intent_by_cluster = Label.label_clusters_by_verbphrases(closest_utterances)

    
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
    graph_builder = graph_builders[args.filter_algorithm]
    graph = graph_builder.create_directed_graph(
        transition_matrix = transition_matrix, 
        intent_by_cluster = intent_by_cluster, 
        tau=args.tau, 
        top_k=args.top_k, 
        alpha = args.alpha 
    )
    dir_name = os.path.join("output", f"dataset{args.file_path}n_clusters_{optimal_cluster_number}n_samples{args.num_sampled_data}tau{args.tau}_top_k{args.top_k}_alpha{args.alpha}")

    # Check if the directory exists, and create it if not
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    graph_builder.plot_graph_html(graph, dir_name , args.filter_algorithm )
    
    # evaluation 
    print("Embedding test data...")
    print(test_data[list(test_data.keys())[0]])
    test_utterances = ExtractEmbed.extract_utterances_test(test_data)
    test_data = ExtractEmbed.embed_sampled_data(test_data, model, dataset_name=args.file_path)
    test_all_embeddings = ExtractEmbed.extract_embeddings(test_data)
    print("Test data embedded")

    print("Assigning test data to clusters...")
    test_data_assigned_cluster_ids = Cluster.assign_to_clusters(cluster_centers, test_all_embeddings)
    print("Test data assigned to clusters.")

    # Evaluate the model
    print("Evaluating the model...")
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
    print(f"Evaluation complete. Scores: { str(scores)}")
    return scores
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quasi-patterned Conversations Analysis")
    parser.add_argument("--file_path" , type = str , default = os.path.join("data" , "ABCD.json") , help="path for json formatted conversations/dialogues" )
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Split ratio between 'train' and 'test' sets")
    parser.add_argument("--num_sampled_data", type=int, default=1000, help="Number of sampled datapoints")
    parser.add_argument("--max_clusters", type=int, default=15, help="Maximum number of clusters for the elbow method")
    parser.add_argument("--min_clusters" ,type=int, default=9, help="Minimum number of clusters for the elbow method" )
    parser.add_argument("--model_name", type=str, default='sentence-transformers/all-MiniLM-L12-v2', help="Model name for SentenceTransformer")
    parser.add_argument("--label_model", type=str, default='open-mixtral-8x22b', help="Model for labeling clusters by closest utterance")
    parser.add_argument("--tau", type=float, default=0.1, help="Minimum weight for conversational graph edges")
    parser.add_argument("--top_k", type=int, default=1, help="Top k edges to keep in the conversational graph")
    parser.add_argument("--alpha", type=float, default=1, help="alpha for adaptive threshold graph builder")
    parser.add_argument("--n_closest" , type=int , default=10, help="Number of the closest utterances to each cluster centroid to be passed to the llm for intent extraction")
    parser.add_argument(
        "--approach",
        type=str,
        default="our_approach",
        choices=["our_approach", "ferreira2024", "carvalho2024"],
        help="The approach to use when applying the graph analysis (choices: our_approach, ferreira2024, carvalho2024)."
    )

    parser.add_argument(
        "--filter_algorithm",
        type=str,
        default="filter_reconnect",
        choices=["adaptive_threshold", "filter_reconnect", "threshold", "top_k"],
        help="The filtering algorithm for the conversational graph (choices: adaptive_threshold, filter_reconnect, threshold, top_k)."
    )
    

    args = parser.parse_args()
    main(args)
