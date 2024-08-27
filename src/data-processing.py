import json
import numpy as np
import random
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx
from dotenv import load_dotenv
import os
from src.llm import CollectionLLM
load_dotenv()


class Data:
    def __init__(self, processed_formatted_data_path: str, number_samples: int, max_clusters: int):
        self.processed_formatted_data = self.read_json_file(processed_formatted_data_path)
        self.number_samples = number_samples
        self.max_clusters = max_clusters
        self.number_clusters = None  # Set this after user input
        self.customer_support_agent_utterances = None
        self.embeddings_sampled_data = None
        self.all_embeddings = None
        self.labels = None
        self.cluster_centers = None
        self.stats = None
        self.transition_matrix = None
        self.markov_chain = None
        self.intent_by_cluster = None 

    def read_json_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Error occurred in read_json_file: {e}")

    def extract_utterances(self):
        customer_support_agent_utterances = {}
        for i, conv in enumerate(self.processed_formatted_data.values()):
            customer_support_agent_utterances[i] = [
                utterance["content"] for utterance in conv if utterance["role"] in ["agent", "action"]
            ]
        self.customer_support_agent_utterances = customer_support_agent_utterances

    def sample_data(self):
        def get_random_n_pairs(input_dict, n):
            if n <= 0:
                return {}
            if n >= len(input_dict):
                return input_dict
            random_keys = random.sample(list(input_dict.keys()), n)
            return {key: input_dict[key] for key in random_keys}

        self.embeddings_sampled_data = get_random_n_pairs(self.customer_support_agent_utterances, self.number_samples)

    def generate_embeddings(self):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        self.data = {}
        for key, sentences in tqdm(self.embeddings_sampled_data.items(), desc="Generating Embeddings"):
            embeddings = model.encode(sentences)
            self.data[key] = [
                {"utterance": sentence, "embedding": embedding.tolist()}
                for sentence, embedding in zip(sentences, embeddings)
            ]
        self.embeddings_sampled_data = data

        self.all_embeddings = []
        for key in self.data:
            for item in self.data[key]:
                self.all_embeddings.append(item["embedding"])
        self.all_embeddings = np.array(self.all_embeddings)

    def determine_optimal_clusters(self):
        sse = []
        for k in range(1, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans.fit(self.all_embeddings)
            sse.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.max_clusters + 1), sse, marker='o')
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('Sum of squared distances (Inertia)')
        plt.show()

        # Ask user for the optimal number of clusters
        self.number_clusters = int(input("Based on the elbow plot, please enter the optimal number of clusters: "))

    def cluster_data(self):
            if self.number_clusters is None:
                print("You need to determine the optimal number of clusters first using the elbow method.")
                return

            # Initial K-means clustering
            kmeans = KMeans(n_clusters=self.number_clusters, init='k-means++', random_state=42)
            self.labels = kmeans.fit_predict(self.all_embeddings)
            self.cluster_centers = kmeans.cluster_centers_
            self.embedding_to_cluster = {i: label for i, label in enumerate(kmeans.labels_)}
            
            # Step 1: Identify and remove outliers
            filtered_embeddings = []
            filtered_labels = []
            
            for cluster_id in range(self.number_clusters):
                # Extract embeddings for the current cluster
                cluster_indices = [i for i, label in enumerate(self.labels) if label == cluster_id]
                cluster_embeddings = self.all_embeddings[cluster_indices]

                # Calculate distances from the centroid
                centroid = self.cluster_centers[cluster_id]
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

                # Determine the 75th percentile distance as the outlier threshold
                distance_threshold = np.percentile(distances, 75)

                # Filter out the outliers (distances greater than the 75th percentile)
                non_outlier_indices = [idx for idx, distance in zip(cluster_indices, distances) if distance <= distance_threshold]

                # Collect the filtered embeddings and labels
                filtered_embeddings.extend(self.all_embeddings[non_outlier_indices])
                filtered_labels.extend([cluster_id] * len(non_outlier_indices))

            # Convert filtered data to numpy arrays
            filtered_embeddings = np.array(filtered_embeddings)

            # Step 2: Reclustering on the filtered data
            kmeans_reclustered = KMeans(n_clusters=self.number_clusters, init='k-means++', random_state=42)
            self.labels = kmeans_reclustered.fit_predict(filtered_embeddings)
            self.cluster_centers = kmeans_reclustered.cluster_centers_

            # Update the embedding-to-cluster mapping after reclustering
            embedding_index = 0
            self.embedding_to_cluster = {}
            for key in self.data:
                for item in self.data[key]:
                    if embedding_index < len(filtered_embeddings):
                        item["cluster"] = self.labels[embedding_index]
                        embedding_index += 1
                    else:
                        item["cluster"] = None  # Handle any leftover data points that may not be in filtered clusters

    def calculate_statistics(self):
        distances = []
        for embedding, label in zip(self.all_embeddings, self.labels):
            distance = np.linalg.norm(embedding - self.cluster_centers[label])
            distances.append(distance)

        distances = np.array(distances)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        percentiles = np.percentile(distances, [25, 50, 75])

        self.stats = {
            "mean": mean_distance,
            "std": std_distance,
            "25,50,75": percentiles
        }

    def plot_clusters(self, plot=False):
        if plot:
            # Ensure we use filtered embeddings and labels
            tsne = TSNE(n_components=2, perplexity=50, random_state=42)
            
            # Apply t-SNE only on the filtered embeddings
            embeddings_2d = tsne.fit_transform(self.all_embeddings)  # Make sure self.all_embeddings is updated after filtering

            colors = [
                "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#800000",
                "#008000", "#000080", "#808000", "#800080", "#008080", "#C0C0C0", "#FFA500",
                "#A52A2A", "#5F9EA0", "#D2691E", "#FF7F50", "#6495ED", "#DC143C", "#00FA9A",
                "#B8860B", "#8B0000", "#E9967A", "#8FBC8F", "#483D8B", "#2F4F4F", "#00CED1",
                "#9400D3", "#FFD700"
            ]
            cmap = ListedColormap(colors)

            # Ensure the number of labels matches the filtered data points
            if len(self.labels) != len(embeddings_2d):
                print("Mismatch between labels and embeddings after filtering. Please check the filtering process.")
                return

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=self.labels, cmap=cmap)
            plt.colorbar(scatter)
            plt.title('t-SNE Visualization of Clusters')
            plt.show()

    def extract_closest_embeddings(self, n=5):
        closest_embeddings = {i: [] for i in range(len(self.cluster_centers))}

        # Calculate the distances of each embedding to its corresponding cluster center
        for idx, (embedding, label) in enumerate(zip(self.all_embeddings, self.labels)):
            distance = np.linalg.norm(embedding - self.cluster_centers[label])
            closest_embeddings[label].append((distance, idx))

        # Sort the distances and extract the n closest embeddings for each cluster
        closest_n_embeddings = {}
        for cluster, distances in closest_embeddings.items():
            sorted_distances = sorted(distances, key=lambda x: x[0])
            closest_n_embeddings[cluster] = sorted_distances[:n]

        # Extract the corresponding utterances
        closest_n_utterances = {}
        for cluster, closest in closest_n_embeddings.items():
            closest_n_utterances[cluster] = []
            for _, idx in closest:
                current_idx = idx
                for key in self.data :
                    if current_idx < len(self.data[key]):
                        closest_n_utterances[cluster].append(self.data[key][current_idx]['utterance'])
                        break
                    else:
                        current_idx -= len(self.data[key])

        return closest_n_utterances

    def label_intents(self, closest_utterances, model, plot=False):
        self.intent_by_cluster = {}
        try : 
            client = CollectionLLM.llm_collection[model]
        except KeyError as e : 
            print(f"model {model} is not available, you need to use one of these models {str(list(CollectionLLM.llm_collection.keys()))} : {e}")
        for cluster, utterances in tqdm(closest_utterances.items(), desc="Labeling Intents"):
            prompt = ("You will be provided with multiple utterances from conversations between a customer support agent and clients from a specific company. Your task is to identify and extract the underlying intent of these utterances. Your response should be a concise phrase summarizing the collective intent, abstracting specific details like names or unique identifiers into generalized categories (e.g., replace 'Alice' with 'Customer Name'). Ensure your output is a single phrase that represents the overarching purpose or request.")

            for utterance in utterances:
                prompt += f"- {utterance}\n"

            prompt += "\nYour response should be a dict with one attribute named 'intent'."

            try:
                completion = client.get_response(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an AI expert in recognizing intent and underlying motives. You will receive utterances from customer support conversations. Your task is to identify and succinctly summarize the intent behind these utterances"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                )
                intent = json.loads(completion.replace("```" , "").replace("json" , "")).get("intent")
                self.intent_by_cluster[str(cluster)] = intent

            except Exception as e:
                print(f"Error reading completion response for cluster {cluster}: {e}")
                intent = str(completion)
                self.intent_by_cluster[str(cluster)] = intent



        return self.intent_by_cluster

    def extract_ordered_intents(self, data, plot=False):
        ordered_intents = []
        for key in data:
            for item in data[key]:
                if "ordered_intents" in item:
                    ordered_intents.append(item["ordered_intents"])

        return ordered_intents

    def create_transition_matrix(self, ordered_intents, intent_by_cluster, plot=False):
        # Create a mapping from intent to cluster number, ignoring "Unknown" intents
        cluster_by_intent = {intent: int(cluster_num) for cluster_num, intent in intent_by_cluster.items() if intent != "Unknown"}

        # Determine the size of the transition matrix based on the number of valid intents
        num_clusters = len(cluster_by_intent)
        transition_matrix = np.zeros((num_clusters, num_clusters))

        # Count transitions while ignoring invalid intents
        for intent_list in ordered_intents:
            for i in range(len(intent_list) - 1):
                current_intent = intent_list[i]
                next_intent = intent_list[i + 1]

                # Only proceed if both intents are valid and present in the mapping
                if current_intent in cluster_by_intent and next_intent in cluster_by_intent:
                    from_idx = cluster_by_intent[current_intent]
                    to_idx = cluster_by_intent[next_intent]

                    # Safeguard against any indexing issues
                    if from_idx < num_clusters and to_idx < num_clusters:
                        transition_matrix[from_idx][to_idx] += 1
                    else:
                        print(f"Skipping out-of-bounds transition: {from_idx} -> {to_idx}")

        # Normalize the counts to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0) * 100

        if plot:
            self.plot_transition_matrix(transition_matrix, {intent: cluster_num for intent, cluster_num in cluster_by_intent.items()})

        return transition_matrix

    def plot_transition_matrix(self, transition_matrix, intent_by_cluster, font_size=8):
        fig, ax = plt.subplots(figsize=(30, 15))
        cax = ax.matshow(transition_matrix, cmap='magma_r')

        plt.title('Transition Matrix', pad=20)
        fig.colorbar(cax)
        ax.set_xticks(np.arange(len(intent_by_cluster)))
        ax.set_yticks(np.arange(len(intent_by_cluster)))
        ax.set_xticklabels(intent_by_cluster.keys(), rotation=90)
        ax.set_yticklabels(intent_by_cluster.keys())

        for (i, j), val in np.ndenumerate(transition_matrix):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=font_size)

        plt.xlabel('To Intent')
        plt.ylabel('From Intent')
        plt.show()
    def add_intents_to_conversations(self):
        for key in self.data:
            # Create an ordered list of intents for each conversation
            intents_ordered = []
            for item in self.data[key]:
                if 'ordered_intents' not in item.keys()  and len(item) != 0  : 
                    cluster = str(item['cluster'])
                    intent = self.intent_by_cluster.get(cluster)  # Handle missing intents
                    intents_ordered.append(intent)
            # Add the ordered list of intents to the conversation
            self.data[key].append({"ordered_intents": intents_ordered})
        return self.data

    def print_datapoints(self , n = 2 , prefix = '')  :
        print(prefix)
        print("=======================")
        for key in self.data.keys():
            print(f"item : {key}")
            for item in self.data[key]:
               print(item)
        print("=======================")

if __name__ == "__main__":
    data = Data("src/data/processed_formatted_conversations.json", number_samples=500, max_clusters=27)
    data.extract_utterances()
    data.sample_data()
    data.generate_embeddings()
    data.determine_optimal_clusters()  # This method now asks the user for the optimal number of clusters
    data.cluster_data()
    data.calculate_statistics()
    data.plot_clusters(plot=True)
    # Extract closest utterances
    n_closest = 10
    closest_utterances = data.extract_closest_embeddings(n=n_closest)
    # Print the closest utterances for each cluster
    for cluster, utterances in closest_utterances.items():
        print(f"Cluster {cluster}:")
        print("=" * 94)
        for idx, utterance in enumerate(utterances):
            print(f"- {utterance}")
        print("=" * 94, "\n\n")

    # Get labeled intents
    intent_by_cluster =data.label_intents(closest_utterances, model="gemini-1.5-flash", plot=True)

    # Print out the intents by cluster
    for cluster, intent in intent_by_cluster.items():
        print(f"Cluster {cluster} Intent: {intent}")
    # Extract ordered intents
    data.add_intents_to_conversations()
    ordered_intents = data.extract_ordered_intents(data.data, plot=True)
    
    # Create and plot transition matrix
    transition_matrix = data.create_transition_matrix(ordered_intents, intent_by_cluster, plot=True)