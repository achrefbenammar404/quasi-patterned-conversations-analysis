import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
from typing import Dict 

colors = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#800000",
    "#008000", "#000080", "#808000", "#800080", "#008080", "#C0C0C0", "#FFA500",
    "#A52A2A", "#5F9EA0", "#D2691E", "#FF7F50", "#6495ED", "#DC143C", "#00FA9A",
    "#B8860B", "#8B0000", "#E9967A", "#8FBC8F", "#483D8B", "#2F4F4F", "#00CED1",
    "#9400D3", "#FFD700"
]
cmap = ListedColormap(colors)

class Cluster : 

    def elbow_method(embeddings, max_clusters=10):
        sse = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans.fit(embeddings)
            sse.append(kmeans.inertia_ / len(embeddings))
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), sse, marker='o')
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('Average Sum of squared distances')
        plt.show()
        
    def cluster_embeddings(data, num_clusters, random_state=42):
        # Extract all embeddings into a single list
        all_embeddings = []
        for key in data:
            for item in data[key]:
                all_embeddings.append(item["embedding"])

        # Convert list to numpy array for clustering
        all_embeddings = np.array(all_embeddings)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=random_state)
        kmeans.fit(all_embeddings)

        # Store cluster results
        # Create a mapping from embedding index to cluster label
        embedding_to_cluster = {i: label for i, label in enumerate(kmeans.labels_)}

        # Update `data` dictionary to include cluster labels
        embedding_index = 0
        for key in data:
            for item in data[key]:
                item["cluster"] = embedding_to_cluster[embedding_index]
                embedding_index += 1

        return data, all_embeddings, kmeans.labels_, kmeans.cluster_centers_
    
    def identify_outliers(embeddings, labels, cluster_centers , percentile ):
        # Calculate the distances of each embedding to its corresponding cluster center
        distances = {i: [] for i in range(len(cluster_centers))}
        for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
            distance = np.linalg.norm(embedding - cluster_centers[label])
            distances[label].append(distance)

        # Determine the 75th percentile distance for each cluster
        percentile_75 = {label: np.percentile(distances[label], percentile) for label in distances}

        # Identify outliers
        outliers = set()
        for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
            distance = np.linalg.norm(embedding - cluster_centers[label])
            if distance > percentile_75[label]:
                outliers.add(idx)

        return outliers
    
    def remove_outliers(data, outliers):
        cleaned_data = {}
        embedding_index = 0
        for key in data:
            cleaned_data[key] = []
            for item in data[key]:
                if embedding_index not in outliers:
                    cleaned_data[key].append(item)
                embedding_index += 1
        return cleaned_data
    
    def visualize_clusters_tsne(embeddings, labels , perplexity = 30 ):
        tsne = TSNE(n_components=2, random_state=42 , perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(20, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels , cmap=cmap)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Clusters')
        plt.show()
        
    def plot_distance_distribution(embeddings, labels, cluster_centers):
        distances = []
        for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
            distance = np.linalg.norm(embedding - cluster_centers[label])
            distances.append(distance)

        distances = np.array(distances)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        percentiles = np.percentile(distances, [25, 50, 75])

        plt.figure(figsize=(12, 8))
        plt.hist(distances, bins=30, alpha=0.7, color='blue', edgecolor='black')

        for percentile, color, label in zip(percentiles, ['green', 'orange', 'red'], ['25th percentile', '50th percentile', '75th percentile']):
            plt.axvline(percentile, color=color, linestyle='dashed', linewidth=2, label=f'{label}: {percentile:.2f}')

        

        plt.xlabel('Distance to Centroid')
        plt.ylabel('Frequency')
        plt.title('Histogram of Distances to Cluster Centroids')
        plt.legend()
        plt.grid(True)
        plt.show()
        return {"mean" : mean_distance , "std" :  std_distance , "25,50,75" : percentiles}
    
    def extract_closest_embeddings(data, embeddings, labels, cluster_centers, n=5):
        closest_embeddings = {i: [] for i in range(len(cluster_centers))}
        
        # Calculate the distances of each embedding to its corresponding cluster center
        for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
            distance = np.linalg.norm(embedding - cluster_centers[label])
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
                for key in data:
                    if current_idx < len(data[key]):
                        closest_n_utterances[cluster].append(data[key][current_idx]['utterance'])
                        break
                    else:
                        current_idx -= len(data[key])

        return closest_n_utterances
    
    def print_closest_utterances(closest_utterances : Dict ) : 
        for cluster, utterances in closest_utterances.items():
            print(f"Cluster {cluster}:")
            print("==============================================================================================")
            for idx , utterance in enumerate(utterances):
                print("- " , utterance)
            print("==============================================================================================\n\n")


        