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

        kmeans = KMeans(n_clusters=self.number_clusters, init='k-means++', random_state=42)
        self.labels = kmeans.fit_predict(self.all_embeddings)
        self.cluster_centers = kmeans.cluster_centers_
        self.embedding_to_cluster = {i: label for i, label in enumerate(kmeans.labels_)}
        
        embedding_index = 0 
        for key in self.data:
            for item in self.data[key]:
                item["cluster"] = self.embedding_to_cluster[embedding_index]
                embedding_index += 1
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
            tsne = TSNE(n_components=2, perplexity=50, random_state=42)
            embeddings_2d = tsne.fit_transform(self.all_embeddings)

            colors = [
                "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#800000",
                "#008000", "#000080", "#808000", "#800080", "#008080", "#C0C0C0", "#FFA500",
                "#A52A2A", "#5F9EA0", "#D2691E", "#FF7F50", "#6495ED", "#DC143C", "#00FA9A",
                "#B8860B", "#8B0000", "#E9967A", "#8FBC8F", "#483D8B", "#2F4F4F", "#00CED1",
                "#9400D3", "#FFD700"
            ]
            cmap = ListedColormap(colors)

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

        for cluster, utterances in tqdm(closest_utterances.items(), desc="Labeling Intents"):
            prompt = ("You will be given some utterances from a conversation between a customer support agent and clients from a specific company. "
                      "You need to extract the intent of these utterances. Your output should be a simple short phrase describing the common overall "
                      "intent of these utterances. Replace any proper names or specifications with the category of the object (for example, replace 'Alice' "
                      "with 'Customer Name').\n")

            for utterance in utterances:
                prompt += f"- {utterance}\n"

            prompt += "\nYour response should be a dict with one attribute named 'intent'."

            try:
                client = OpenAI(base_url=os.getenv("BASE_URL_OLLAMA"), api_key=os.getenv('API_KEY_OLLAMA'))
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an AI specializing in intent and motive recognition. You will be given utterances from customer support agent conversations and output intent in JSON format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                )
                intent = json.loads(completion.choices[0].message.content).get("intent", "Unknown")
                self.intent_by_cluster[str(cluster)] = intent

            except Exception as e:
                print(f"Error reading completion response for cluster {cluster}: {e}")
                intent = str(completion.choices[0].message.content)
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
        cluster_by_intent = {intent: int(cluster_num) for cluster_num, intent in intent_by_cluster.items()}
        # Initialize the transition matrix with zeros
        transition_matrix = np.zeros((len(intent_by_cluster), len(intent_by_cluster)))
            # Count transitions
        for intent_list in ordered_intents:
            for i in range(len(intent_list) - 1):
                current_intent = intent_list[i]
                next_intent = intent_list[i + 1]
                transition_matrix[cluster_by_intent[current_intent]][cluster_by_intent[next_intent]] += 1

        # Normalize the counts to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0) * 100

        if plot:
            self.plot_transition_matrix(transition_matrix, intent_by_cluster)

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
                    intent = self.intent_by_cluster.get(cluster, "Unknown")  # Handle missing intents
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
    data = Data("src/data/processed_formatted_conversations.json", number_samples=500, max_clusters=40)
    data.extract_utterances()
    data.sample_data()
    data.generate_embeddings()
    data.determine_optimal_clusters()  # This method now asks the user for the optimal number of clusters
    data.cluster_data()
    data.calculate_statistics()
    data.plot_clusters(plot=True)
    # Extract closest utterances
    n_closest = 5
    closest_utterances = data.extract_closest_embeddings(n=n_closest)
    # Print the closest utterances for each cluster
    for cluster, utterances in closest_utterances.items():
        print(f"Cluster {cluster}:")
        print("=" * 94)
        for idx, utterance in enumerate(utterances):
            print(f"- {utterance}")
        print("=" * 94, "\n\n")

    # Get labeled intents
    intent_by_cluster = data.label_intents(closest_utterances, model="phi", plot=True)

    # Print out the intents by cluster
    for cluster, intent in intent_by_cluster.items():
        print(f"Cluster {cluster} Intent: {intent}")
    # Extract ordered intents
    data.add_intents_to_conversations()
    ordered_intents = data.extract_ordered_intents(data.data, plot=True)
    
    # Create and plot transition matrix
    transition_matrix = data.create_transition_matrix(ordered_intents, intent_by_cluster, plot=True)