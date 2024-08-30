from src.clustering import Cluster
from src.embedding import ExtractEmbed
from src.label import Label
from src.transition_matrix import TransitionAnalysis
from src.utils.utils import * 
from sentence_transformers import SentenceTransformer
from src.MarkovChain import ConversationalGraph

def main() : 
    conversations = read_json_file("data/processed_formatted_conversations.json")
    customer_support_utterances = ExtractEmbed.extract_customer_support_utterances(conversations)
    sampled_data = get_random_n_pairs(customer_support_utterances , 1000)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    
    data = ExtractEmbed.embed_sampled_data(sampled_data , model )
    
    all_embeddings = ExtractEmbed.extract_embeddings(data)
    
    Cluster.elbow_method(all_embeddings ,max_clusters = 30 )
    
    optimal_cluster_number = int(input("optimal cluster number : "))
    clustered_data, embeddings , labels , cluster_centers = Cluster.cluster_embeddings(data, num_clusters=optimal_cluster_number)
    Cluster.visualize_clusters_tsne(embeddings , labels , perplexity=50)
    
    stats = Cluster.plot_distance_distribution(embeddings , labels , cluster_centers)
    print(stats)
    outliers = Cluster.identify_outliers(embeddings, labels, cluster_centers, 75)

    cleaned_data = Cluster.remove_outliers(clustered_data , outliers)
    reclustered_data, cleaned_embeddings, cleaned_labels, cleaned_cluster_centers = Cluster.cluster_embeddings(cleaned_data, num_clusters=optimal_cluster_number)
    
    Cluster.visualize_clusters_tsne(cleaned_embeddings, cleaned_labels , perplexity=50)
    
    Cluster.plot_distance_distribution(cleaned_embeddings, cleaned_labels, cluster_centers)
    
    n_closest = 2
    closest_utterances = Cluster.extract_closest_embeddings(reclustered_data, cleaned_embeddings, cleaned_labels, cluster_centers, n=n_closest)
    Cluster.print_closest_utterances(closest_utterances)
    
    
    intent_by_cluster  = Label.label_clusters_by_closest_utterances(closest_utterances , model  = "open-mixtral-8x22b")


    cluster_by_intent = Label.generate_cluster_by_intent(intent_by_cluster)
    
    updated_data_with_intents = Label.add_intents_to_conversations(reclustered_data , intent_by_cluster)
    
    Label.print_updated_data_with_ordered_intents(updated_data_with_intents)
    
    ordered_intents = Label.extract_ordered_intents(updated_data_with_intents)
    
    transition_matrix = TransitionAnalysis.create_transition_matrix(ordered_intents , intent_by_cluster)
    
    TransitionAnalysis.plot_transition_matrix(transition_matrix, intent_by_cluster)
    
    graph = ConversationalGraph.create_directed_graph(transition_matrix , intent_by_cluster, min_weight = 0.1 , algorithm = "filter&reconnect", top_k = 1)
    ConversationalGraph.plot_graph_html(graph , "filter&reconnect")
    
    



if __name__ == "__main__" : 
    main()