import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Dict, List, Any, Union, Set
import os

colors = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#800000",
    "#008000", "#000080", "#808000", "#800080", "#008080", "#C0C0C0", "#FFA500",
    "#A52A2A", "#5F9EA0", "#D2691E", "#FF7F50", "#6495ED", "#DC143C", "#00FA9A",
    "#B8860B", "#8B0000", "#E9967A", "#8FBC8F", "#483D8B", "#2F4F4F", "#00CED1",
    "#9400D3", "#FFD700"
]
cmap = ListedColormap(colors)

class Cluster:

    def elbow_method(embeddings: List[np.array], min_clusters : int = 5 , max_clusters: int = 10, 
                     title: str = 'Elbow Method For Optimal k', 
                     dir_path: str = "output") -> None:
        """
        Performs the elbow method to determine the optimal number of clusters 
        for KMeans by plotting the average sum of squared distances against 
        the number of clusters.

        Args:
            embeddings (List[np.array]): List of embeddings.
            max_clusters (int, optional): Maximum number of clusters. Defaults to 10.
            title (str, optional): Title of the plot. Defaults to 'Elbow Method For Optimal k'.
            dir_path (str, optional): Directory path where the plot will be saved. Defaults to "output".
        """
        os.makedirs(dir_path, exist_ok=True)

        sse = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans.fit(embeddings)
            sse.append(kmeans.inertia_ / len(embeddings))
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), sse, marker='o')
        plt.title(title)
        plt.xlabel('Number of clusters')
        plt.ylabel('Average Sum of Squared Distances')
        file_path = os.path.join(dir_path, title.replace(' ', '_') + '.png')
        plt.savefig(file_path, format='png', dpi=300)
        plt.show()
        plt.close()

    def cluster_embeddings(data: Dict[str, List[Dict[str, Any]]], num_clusters: int, 
                           random_state: int = 42) -> Union[Dict, np.array, List[int], np.array]:
        """
        Clusters the embeddings in the provided data using KMeans.

        Args:
            data (Dict[str, List[Dict[str, Any]]]): Dictionary with `conv-id`s as keys and 
                lists of dictionaries with `embedding` and `utterance` as attributes.
            num_clusters (int): Number of clusters.
            random_state (int, optional): Random state for KMeans algorithm. Defaults to 42.

        Returns:
            Tuple[Dict, np.array, List[int], np.array]: Updated data dictionary with cluster labels, 
                all embeddings, cluster labels, and cluster centers.
        """
        all_embeddings = [item["embedding"] for key in data for item in data[key]]
        all_embeddings = np.array(all_embeddings)

        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=random_state)
        kmeans.fit(all_embeddings)

        embedding_to_cluster = {i: label for i, label in enumerate(kmeans.labels_)}

        embedding_index = 0
        for key in data:
            for item in data[key]:
                item["cluster"] = embedding_to_cluster[embedding_index]
                embedding_index += 1

        return data, all_embeddings, kmeans.labels_, kmeans.cluster_centers_

   
    def visualize_clusters_tsne(embeddings: List[Union[np.array , List[float]]], labels: List[int], 
                                perplexity: int = 30, title: str = 't-SNE Visualization of Clusters', 
                                dir_path: str = "output/") -> None:
        """
        Visualizes clusters using t-SNE.

        Args:
            embeddings (np.array): Numpy array of embeddings.
            labels (List[int]): List of cluster labels.
            perplexity (int, optional): Perplexity parameter for t-SNE. Defaults to 30.
            title (str, optional): Title of the plot. Defaults to 't-SNE Visualization of Clusters'.
            dir_path (str, optional): Directory path where the plot will be saved. Defaults to "output/".
        """
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
        os.makedirs(dir_path, exist_ok=True)

        plt.figure(figsize=(20, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap=cmap)
        plt.colorbar(scatter)
        plt.title(title)
        file_path = os.path.join(dir_path, title.replace(' ', '_') + '.png')
        plt.savefig(file_path, format='png', dpi=300)
        plt.close()

    def plot_distance_distribution(embeddings: np.array, labels: List[int], 
                                   cluster_centers: np.array, title: str = 'Histogram of Distances to Cluster Centroids', 
                                   dir_path: str = "output/") -> Dict[str, Union[float, np.array]]:
        """
        Plots the distribution of distances from each embedding to its corresponding cluster centroid.

        Args:
            embeddings (np.array): Numpy array of embeddings.
            labels (List[int]): List of cluster labels.
            cluster_centers (np.array): Numpy array of cluster centroids.
            title (str, optional): Title of the plot. Defaults to 'Histogram of Distances to Cluster Centroids'.
            dir_path (str, optional): Directory path where the plot will be saved. Defaults to "output/".

        Returns:
            Dict[str, Union[float, np.array]]: Dictionary containing mean, standard deviation, and 
                25th, 50th, and 75th percentiles of distances.
        """
        distances = [np.linalg.norm(embedding - cluster_centers[label]) for embedding, label in zip(embeddings, labels)]

        distances = np.array(distances)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        percentiles = np.percentile(distances, [25, 50, 75])

        plt.figure(figsize=(12, 8))
        plt.hist(distances, bins=30, alpha=0.7, color='blue', edgecolor='black')

        for percentile, color, label in zip(percentiles, ['green', 'orange', 'red'], 
                                            ['25th percentile', '50th percentile', '75th percentile']):
            plt.axvline(percentile, color=color, linestyle='dashed', linewidth=2, label=f'{label}: {percentile:.2f}')

        plt.xlabel('Distance to Centroid')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.grid(True)

        os.makedirs(dir_path, exist_ok=True)

        file_path = os.path.join(dir_path, title.replace(' ', '_') + '.png')
        plt.savefig(file_path, format='png', dpi=300)
        plt.close()

        return {"mean": mean_distance, "std": std_distance, "25,50,75": percentiles}

    def extract_closest_embeddings(data: Dict[str, List[Dict[str, Any]]], 
                                   embeddings: np.array, labels: List[int], 
                                   cluster_centers: np.array, n: int = 5) -> Dict[int, List[str]]:
        """
        Extracts the closest n embeddings to each cluster center.

        Args:
            data (Dict[str, List[Dict[str, Any]]]): Data dictionary with `conv-id`s as keys and 
                lists of dictionaries with `embedding` and `utterance` attributes.
            embeddings (np.array): Numpy array of embeddings.
            labels (List[int]): List of cluster labels.
            cluster_centers (np.array): Numpy array of cluster centroids.
            n (int, optional): Number of closest embeddings to extract. Defaults to 5.

        Returns:
            Dict[int, List[str]]: Dictionary mapping cluster indices to lists of closest utterances.
        """
        closest_embeddings = {i: [] for i in range(len(cluster_centers))}
        
        for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
            distance = np.linalg.norm(embedding - cluster_centers[label])
            closest_embeddings[label].append((distance, idx))

        closest_n_embeddings = {cluster: sorted(distances, key=lambda x: x[0])[:n] 
                                for cluster, distances in closest_embeddings.items()}

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

    def print_closest_utterances(closest_utterances: Dict[int, List[str]]) -> None:
        """
        Prints the closest utterances to each cluster center.

        Args:
            closest_utterances (Dict[int, List[str]]): Dictionary mapping cluster indices 
                to lists of closest utterances.
        """
        for cluster, utterances in closest_utterances.items():
            print(f"Cluster {cluster}:")
            print("=" * 94)
            for idx, utterance in enumerate(utterances):
                print(f"- {utterance}")
            print("=" * 94 + "\n\n")
