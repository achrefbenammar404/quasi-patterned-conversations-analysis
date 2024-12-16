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
from src.evaluation import (
    SemanticEvaluator,
    StructuralEvaluator
)

graph_builders  : Dict[str , ConversationalGraphBuilder ]= {
    'adaptive_threshold_graph_builder' : AdaptiveThresholdGraphBuilder , 
    'filter_reconnect_graph_builder' : FilterReconnectGraphBuilder , 
    'threshold_graph_builder' : ThresholdGraphBuilder , 
    'top_k_graph_builder' : TopKGraphBuilder
}



def main( dataset_name : str , train_data , test_data , min_clusters , max_clusters , model : SentenceTransformer , alpha , tau , top_k , n_closest , label_model):
    

    
    # Embed the sampled data
    data = ExtractEmbed.embed_sampled_data(train_data, model , dataset_name = dataset_name)
    
    # Extract embeddings
    all_embeddings = ExtractEmbed.extract_embeddings(data)
    
    if min_clusters !=  max_clusters : 
        # Use the elbow method to determine optimal number of clusters
        Cluster.elbow_method(all_embeddings, min_clusters = min_clusters , max_clusters=max_clusters)
        
    # Ask for the optimal cluster number
        optimal_cluster_number = int(input("Optimal cluster number: "))
    else : 
        optimal_cluster_number = min_clusters 
    
    # Cluster the embeddings
    clustered_data, embeddings, labels, cluster_centers = Cluster.cluster_embeddings(data, num_clusters=optimal_cluster_number)
    
    # Extract closest utterances
    n_closest = n_closest
    closest_utterances = Cluster.extract_closest_embeddings(clustered_data, embeddings, labels, cluster_centers, n=n_closest)
    Cluster.print_closest_utterances(closest_utterances)
    
    # Label clusters
    intent_by_cluster = Label.label_clusters_by_closest_utterances(closest_utterances, model=label_model)
    

    
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
    graphs = {}
    for builder_name , builder in graph_builders.items():
        graphs[builder_name] = builder.create_directed_graph(
            transition_matrix = transition_matrix, 
            intent_by_cluster = intent_by_cluster, 
            tau=tau, 
            top_k=top_k, 
            alpha = alpha 
        )
    
    structural_scores = {builder_name : StructuralEvaluator.evaluate(graph) for builder_name , graph in graphs.items() }
    
    
    test_utterances = ExtractEmbed.extract_utterances(test_data)
    # Embed the sampled data
    test_data = ExtractEmbed.embed_sampled_data(test_data, model , dataset_name = dataset_name)
    
    # Extract embeddings
    test_all_embeddings = ExtractEmbed.extract_embeddings(test_data)
    test_data_assigned_cluster_ids = Cluster.assign_to_clusters(cluster_centers , test_all_embeddings)
    test_ordered_intents = []
    counter = 0 
    for conv in test_utterances : 
        intents = []
        for utterance in conv : 
            intents.append(intent_by_cluster[str(test_data_assigned_cluster_ids[counter])])
            counter+=1
        test_ordered_intents.append(intents)  
    print(test_ordered_intents)
    semantic_scores = {
        builder_name : SemanticEvaluator.evaluate(
            graphs[builder_name] , test_ordered_intents , test_utterances , model
        )
    } 
    scores ={
        "semantic_scores" : semantic_scores , 
        "structural_scores" : structural_scores
    }
    return graphs , scores 


