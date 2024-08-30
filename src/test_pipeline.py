# import json
# def read_json_file(file_path) : 
#     try : 
#         with open(file_path, 'r') as file:
#             data = json.load(file)
#         return data
#     except Exception as e : 
#         print(f"error occured in read_json_file  : {e}")

# conversations = read_json_file("data/processed_formatted_conversations.json")
print(conversations["0"])


# replaced by ExtractEmbed.extract_customer_support_utterabces (processed_formatted_conversations):  
# customer_support_agent_utterances = {}
# for i , conv in enumerate(conversations.values()) : 
#     customer_support_agent_utterances[i] = [
#         utterance["content"] for utterance in conv if utterance["role"] in["agent" , "action"]
#     ]
    
print(customer_support_agent_utterances[0])

import random
import numpy as np 


# def get_random_n_pairs(input_dict, n):
#     """
#     Get n random key-value pairs from a dictionary.

#     Parameters:
#     input_dict (dict): The input dictionary.
#     n (int): The number of key-value pairs to return.

#     Returns:
#     dict: A dictionary containing n random key-value pairs.
#     """
#     if n <= 0:
#         return {}

#     if n >= len(input_dict):
#         return input_dict

#     # Get n random keys from the dictionary
#     random_keys = random.sample(list(input_dict.keys()), n)

#     # Create a new dictionary with the random keys and their corresponding values
#     random_pairs = {key: input_dict[key] for key in random_keys}

#     return random_pairs




sampled_data = get_random_n_pairs(customer_support_agent_utterances , 100)
print(sampled_data)

#replaced by ExtractEmbed.embed_sentences
# from sentence_transformers import SentenceTransformer


# model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
# def embed_sentences(sentences):
#     """
#     This function takes in a list of sentences and returns their embeddings using the OpenAI Ada-002 model.

#     :param sentences: List of sentences to be embedded
#     :return: List of embeddings for each sentence
#     """
#     # Make the API call to get the embeddings
#     embeddings = model.encode(sentences=sentences)
    
    
#     return embeddings



#replaced by ExtractEmbed.embed_sampled_data 
# from tqdm import tqdm 

# total_keys = len(sampled_data.keys())

# data = {}
# for i, key in tqdm(enumerate(sampled_data.keys(), 1) , desc = "embedding in progress ..."):
#     sentences = sampled_data[key]
#     embeddings = embed_sentences(sentences=sentences)
#     data[key] = [
#         {"utterance": sentence, "embedding": list(embedding)}
#         for sentence, embedding in zip(sentences, embeddings)
#     ]



import json
import numpy as np

# def convert_numpy_int_to_python(data):
#     if isinstance(data, dict):
#         return {key: convert_numpy_int_to_python(value) for key, value in data.items()}
#     elif isinstance(data, list):
#         return [convert_numpy_int_to_python(element) for element in data]
#     elif isinstance(data, np.integer):
#         return int(data)
#     elif isinstance(data, np.floating):
#         return float(data)
#     else:
#         return data

# def save_dict_json(path, data):
#     try:
#         # Convert numpy int32 to Python int
#         data = convert_numpy_int_to_python(data)
#         with open(path, "w") as f:
#             json.dump(data, f, indent=5)
#     except Exception as e:
#         print(f"error occurred in save_dict_json: {e}")
    
keys = list(data.keys())
clustering_keys , test_keys = keys[: int(0.8 * len(keys)) ] ,  keys[int(0.8 * len(keys)) : ]
clustering_data = {}
test_data = {}
for key in clustering_keys : 
    clustering_data[key] = data[key]
for key in test_keys  : 
    test_data[key] = data[key]
data = None 

save_dict_json( "clustering_data.json", clustering_data )
save_dict_json("test_data.json" , test_data)

clustering_data = read_json_file("clustering_data.json")
test_data = read_json_file("test_data.json")

for  key , value in clustering_data.items() : 
    clustering_data[key] = [
        {'utterance' : e['utterance'] , 'embedding' : e['embedding'] } 
        for e in value 
    ]
    
data = None 
sampled_data = None 
conversations = None 

#replaced by methods in class Cluster 
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from sklearn.metrics import silhouette_score
# from matplotlib.colors import ListedColormap
# from scipy.stats import gaussian_kde

# colors = [
#     "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#800000",
#     "#008000", "#000080", "#808000", "#800080", "#008080", "#C0C0C0", "#FFA500",
#     "#A52A2A", "#5F9EA0", "#D2691E", "#FF7F50", "#6495ED", "#DC143C", "#00FA9A",
#     "#B8860B", "#8B0000", "#E9967A", "#8FBC8F", "#483D8B", "#2F4F4F", "#00CED1",
#     "#9400D3", "#FFD700"
# ]
# # Create the ListedColormap
# cmap = ListedColormap(colors)

# def elbow_method(embeddings, max_clusters=10):
#     sse = []
#     for k in range(1, max_clusters + 1):
#         kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
#         kmeans.fit(embeddings)
#         sse.append(kmeans.inertia_ / len(embeddings))
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, max_clusters + 1), sse, marker='o')
#     plt.title('Elbow Method For Optimal k')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Average Sum of squared distances')
#     plt.show()


# #data is the output of the EXtractEmbed.embed_sampled_data 
# def cluster_embeddings(data, num_clusters, random_state=42):
#     # Extract all embeddings into a single list
#     all_embeddings = []
#     for key in data:
#         for item in data[key]:
#             all_embeddings.append(item["embedding"])

#     # Convert list to numpy array for clustering
#     all_embeddings = np.array(all_embeddings)

#     # Apply K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=random_state)
#     kmeans.fit(all_embeddings)

#     # Store cluster results
#     # Create a mapping from embedding index to cluster label
#     embedding_to_cluster = {i: label for i, label in enumerate(kmeans.labels_)}

#     # Update `data` dictionary to include cluster labels
#     embedding_index = 0
#     for key in data:
#         for item in data[key]:
#             item["cluster"] = embedding_to_cluster[embedding_index]
#             embedding_index += 1

#     return data, all_embeddings, kmeans.labels_, kmeans.cluster_centers_

# def identify_outliers(embeddings, labels, cluster_centers , percentile ):
#     # Calculate the distances of each embedding to its corresponding cluster center
#     distances = {i: [] for i in range(len(cluster_centers))}
#     for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
#         distance = np.linalg.norm(embedding - cluster_centers[label])
#         distances[label].append(distance)

#     # Determine the 75th percentile distance for each cluster
#     percentile_75 = {label: np.percentile(distances[label], percentile) for label in distances}

#     # Identify outliers
#     outliers = set()
#     for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
#         distance = np.linalg.norm(embedding - cluster_centers[label])
#         if distance > percentile_75[label]:
#             outliers.add(idx)

#     return outliers
# def remove_outliers(data, outliers):
#     cleaned_data = {}
#     embedding_index = 0
#     for key in data:
#         cleaned_data[key] = []
#         for item in data[key]:
#             if embedding_index not in outliers:
#                 cleaned_data[key].append(item)
#             embedding_index += 1
#     return cleaned_data

# def visualize_clusters_tsne(embeddings, labels , perplexity = 30 ):
#     tsne = TSNE(n_components=2, random_state=42 , perplexity=perplexity)
#     embeddings_2d = tsne.fit_transform(embeddings)

#     plt.figure(figsize=(20, 10))
#     scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels , cmap=cmap)
#     plt.colorbar(scatter)
#     plt.title('t-SNE Visualization of Clusters')
#     plt.show()
    
# def plot_distance_distribution(embeddings, labels, cluster_centers):
#     distances = []
#     for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
#         distance = np.linalg.norm(embedding - cluster_centers[label])
#         distances.append(distance)

#     distances = np.array(distances)
#     mean_distance = np.mean(distances)
#     std_distance = np.std(distances)
#     percentiles = np.percentile(distances, [25, 50, 75])

#     plt.figure(figsize=(12, 8))
#     plt.hist(distances, bins=30, alpha=0.7, color='blue', edgecolor='black')

#     for percentile, color, label in zip(percentiles, ['green', 'orange', 'red'], ['25th percentile', '50th percentile', '75th percentile']):
#         plt.axvline(percentile, color=color, linestyle='dashed', linewidth=2, label=f'{label}: {percentile:.2f}')

    

#     plt.xlabel('Distance to Centroid')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of Distances to Cluster Centroids')
#     plt.legend()
#     plt.grid(True)
#     #plt.show()
    
#     return {"mean" : mean_distance , "std" :  std_distance , "25,50,75" : percentiles}


data = read_json_file("clustering_data.json")

#replaced by ExtractEmbed.extract_embeddings 
# # Extract embeddings
# all_embeddings = []
# for key in data:
#     for item in data[key]:
#         all_embeddings.append(item["embedding"])

all_embeddings = np.array(all_embeddings)


# Step 1: Determine the optimal number of clusters using the elbow method
elbow_method(all_embeddings , max_clusters=30)

# Based on the elbow plot, decide on the number of clusters, e.g., 3
optimal_clusters = 29

# Step 2: Initial clustering
clustered_data, embeddings , labels , cluster_centers = cluster_embeddings(data, num_clusters=optimal_clusters)

visualize_clusters_tsne(embeddings , labels , perplexity=50)


stat = plot_distance_distribution(embeddings , labels , cluster_centers)

print(stat)

# Step 3: Identify outliers
outliers = identify_outliers(embeddings, labels, cluster_centers, 50)


# Step 4: Remove outliers from data
cleaned_data = remove_outliers(clustered_data, outliers)

# Step 5: Re-cluster the cleaned data
reclustered_data, cleaned_embeddings, cleaned_labels, cleaned_cluster_centers = cluster_embeddings(cleaned_data, num_clusters=optimal_clusters)


# Step 6: Visualize the clusters using t-SNE
visualize_clusters_tsne(cleaned_embeddings, cleaned_labels , perplexity=50)

plot_distance_distribution(cleaned_embeddings, cleaned_labels, cluster_centers)

# replaced by Cluster.extract_closest_embedding
# def extract_closest_embeddings(data, embeddings, labels, cluster_centers, n=5):
#     closest_embeddings = {i: [] for i in range(len(cluster_centers))}
    
#     # Calculate the distances of each embedding to its corresponding cluster center
#     for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
#         distance = np.linalg.norm(embedding - cluster_centers[label])
#         closest_embeddings[label].append((distance, idx))

#     # Sort the distances and extract the n closest embeddings for each cluster
#     closest_n_embeddings = {}
#     for cluster, distances in closest_embeddings.items():
#         sorted_distances = sorted(distances, key=lambda x: x[0])
#         closest_n_embeddings[cluster] = sorted_distances[:n]

#     # Extract the corresponding utterances
#     closest_n_utterances = {}
#     for cluster, closest in closest_n_embeddings.items():
#         closest_n_utterances[cluster] = []
#         for _, idx in closest:
#             current_idx = idx
#             for key in data:
#                 if current_idx < len(data[key]):
#                     closest_n_utterances[cluster].append(data[key][current_idx]['utterance'])
#                     break
#                 else:
#                     current_idx -= len(data[key])

#     return closest_n_utterances


n_closest = 100
closest_utterances = extract_closest_embeddings(reclustered_data, cleaned_embeddings, cleaned_labels, cluster_centers, n=n_closest)


#replaced by Cluster.print_closest_utterances 
# # Print the closest utterances for each cluster
# for cluster, utterances in closest_utterances.items():
#     print(f"Cluster {cluster}:")
#     print("==============================================================================================")
#     for idx , utterance in enumerate(utterances):
#         print("- " , utterance)
#     print("==============================================================================================\n\n")


from src.llm import MistralLLM
client = MistralLLM("open-mixtral-8x22b")

# replaced by Label.label_clusters_by_closest_utterances 
# intent_by_cluster = {}
# for cluster, utterances in closest_utterances.items():
#     prompt = f"you will be given some utterances from a conversation between a customer support agent and clients from a specific company. you need to extract the intent of these utterances, your output is a simple short phrase describing the common overall intent of these utterances , replace any proper name or specification with the category of the object (for example when given the name Alice , you replace it with 'Customer Name'): \n"
#     for idx , utterance in enumerate(utterances):
#         prompt += "- " + utterance
#     prompt += "\n your response should be a dict with one attribute that is 'intent' "
#     completion = client.get_response(
#         messages=[
#             {"role": "system", "content": "You are an AI agent specializing in intent and motive recoginition , you will be given utterances of a customer support agent from conversations with clients designed to output a json with one key 'intent'"},
#             {"role": "user", "content": prompt}
#         ]

#     )
#     completion  = completion.replace("```json" , "").replace("```" , "")
#     try : 
#         intent_by_cluster[str(cluster)] = json.loads(completion)["intent"]
#     except Exception as e : 
#         print(f"error reading completion response")
#         intent_by_cluster[str(cluster)] = completion
    
    
print(json.dumps(intent_by_cluster , indent=4))
save_dict_json("intent_by_cluster.json" , intent_by_cluster)

# replaced by Label.generate_cluster_by_intent method 
# cluster_by_intent  = { value : key for key , value in intent_by_cluster.items()}

#print(reclustered_data[list(reclustered_data.keys())[0]][0]['cluster'])
print(reclustered_data[list(reclustered_data.keys())[0]])


#replaced by Label.add_intents_to_conversations 
# def add_intents_to_conversations(data, intents_by_cluster):
#     for key in data:
#         # Create an ordered list of intents for each conversation
#         intents_ordered = []
#         for item in data[key]:
#             if 'ordered_intents' not in item.keys()  and len(item) != 0  : 
#                 cluster = str(item['cluster'])
#                 intent = intents_by_cluster.get(cluster, "Unknown")  # Handle missing intents
#                 intents_ordered.append(intent)
#         # Add the ordered list of intents to the conversation
#         data[key].append({"ordered_intents": intents_ordered})
#     return data
# Add intents to the reclustered data
updated_data_with_intents = add_intents_to_conversations(reclustered_data, intent_by_cluster)

# Print the updated data with ordered intents
#replaced by Label.print_updated_data_with_ordered_intents
# for key in updated_data_with_intents:
#     print(f"Conversation: {key}")
#     for item in updated_data_with_intents[key]:
#         if "utterance" in item:
#             print(f"  Utterance: {item['utterance']}, Cluster: {item['cluster']}")
#         else:
#             print(f"  Ordered Intents: {item['ordered_intents']}")
            
            
save_dict_json( "updated_data_with_intents.json", updated_data_with_intents )
updated_data_with_intents = read_json_file("updated_data_with_intents.json")


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

#replaced by Label.extract_ordered_intents

# def extract_ordered_intents(data):
#     ordered_intents = []
#     for key in data:
#         for item in data[key]:
#             if "ordered_intents" in item:
#                 ordered_intents.append(item["ordered_intents"])
#     return ordered_intents


#replaced by TransitionAnalysis.create_transition_matrix 
# def create_transition_matrix(ordered_intents , intent_by_cluster):
    
#     cluster_by_intent = {intent : int(cluster_num) for cluster_num , intent in intent_by_cluster.items()}
#     # Initialize the transition matrix with zeros
#     transition_matrix = np.zeros((len(intent_by_cluster), len(intent_by_cluster)))
    
#     # Count transitions
#     for intent_list in ordered_intents:
#         for i in range(len(intent_list)-1 ): 
#             current_intent = intent_list[i]
#             next_intent = intent_list[i + 1]
#             transition_matrix[cluster_by_intent[current_intent]][cluster_by_intent[next_intent]] += 1

#     # Normalize the counts to probabilities
#     row_sums = transition_matrix.sum(axis=1, keepdims=True) 
#     transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0) * 100 
    
#     return transition_matrix

# def apply_str_filter(s) : 
#     return [''.join([w[0].upper() for w in sentence.split(" ")])  for sentence in s   ]

#reaplced  transition Analysis 

# def plot_transition_matrix(transition_matrix, intent_by_cluster, font_size=8):
#     fig, ax = plt.subplots(figsize=(100, 50))
#     cax = ax.matshow(transition_matrix, cmap='magma_r')

#     plt.title('Transition Matrix', pad=20)
#     fig.colorbar(cax)
#     ax.set_xticks(np.arange(len(intent_by_cluster)))
#     ax.set_yticks(np.arange(len(intent_by_cluster)))
#     ax.set_xticklabels(intent_by_cluster.values(), rotation=90)
#     ax.set_yticklabels(intent_by_cluster.values())

#     for (i, j), val in np.ndenumerate(transition_matrix):
#         ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=font_size)

#     plt.xlabel('To Intent')
#     plt.ylabel('From Intent')
#     plt.show()

ordered_intents = extract_ordered_intents(updated_data_with_intents)

transition_matrix = create_transition_matrix(ordered_intents , intent_by_cluster)

# Plot transition matrix
plot_transition_matrix(transition_matrix, intent_by_cluster)
import numpy as np

# def save_transition_matrix(transition_matrix, filename='transition_matrix_ada.npy'):
#     np.save(filename, transition_matrix)
# def read_transition_matrix(filename='transition_matrix_ada.npy'):
#     return np.load(filename)


save_transition_matrix(transition_matrix , "transition_matrix_ada.npy")
transition_matrix = read_transition_matrix("transition_matrix_ada.npy")


import numpy as np
import matplotlib.pyplot as plt


# replaced by TransitionAnalysis 
# def plot_scaled_probability_distribution(transition_matrix):
#     # Flatten the transition matrix and remove zero entries
#     scaled_probabilities = transition_matrix.flatten()
#     scaled_probabilities = scaled_probabilities[scaled_probabilities > 0]

#     # Calculate complementary percentiles
#     thresholds = np.percentile(scaled_probabilities, [25, 50, 75])

#     # Define bins in log scale
#     bins = np.logspace(np.log10(scaled_probabilities.min()), np.log10(scaled_probabilities.max()), 30)

#     plt.figure(figsize=(20, 15))
#     plt.hist(scaled_probabilities, bins=bins, alpha=0.75, color='blue', edgecolor='black')
#     plt.xscale('log')  # Set x-axis to log scale

#     # Add percentile lines
#     for threshold, color, label in zip(thresholds, ['red', 'orange', 'green'], ['25th percentile', '50th percentile', '75th percentile']):
#         plt.axvline(threshold, color=color, linestyle='dashed', linewidth=2, label=f'{label}: {threshold:.2f}')

#     plt.title('Probability Distribution of Transitions (Log Scale)')
#     plt.xlabel('Scaled Transition Probability (Log Scale)')
#     plt.ylabel('Frequency')
#     plt.legend()

#     # Set custom x-ticks
#     x_ticks = np.logspace(np.log10(scaled_probabilities.min()), np.log10(scaled_probabilities.max()), 10)
#     plt.xticks(x_ticks, labels=[f'{tick:.2f}' for tick in x_ticks], rotation=90)
    
#     # Add grid lines
#     plt.grid(True, which="both", ls="--", linewidth=0.5)

#     plt.show()
    
#     return thresholds

# Example usage
ordered_intents = extract_ordered_intents(updated_data_with_intents)
transition_matrix = create_transition_matrix(ordered_intents, intent_by_cluster)

# Plot scaled probability distribution and get the complementary percentiles
complementary_percentiles = plot_scaled_probability_distribution(transition_matrix)

# Print the complementary percentiles
print(f"25% of scaled transition probabilities are less than: {complementary_percentiles[0]:.2f}")
print(f"50% of scaled transition probabilities are less than: {complementary_percentiles[1]:.2f}")
print(f"75% of scaled transition probabilities are less than: {complementary_percentiles[2]:.2f}")
