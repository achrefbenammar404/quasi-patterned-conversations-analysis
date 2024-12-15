import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict

class TransitionAnalysis:

    def create_transition_matrix(ordered_intents: List[List[str]], 
                                 intent_by_cluster: Dict[str, str]) -> np.ndarray:
        """
        Creates a transition matrix based on ordered intents and their corresponding clusters.

        Args:
            ordered_intents (List[List[str]]): List of lists, where each inner list contains a sequence of intents.
            intent_by_cluster (Dict[str, str]): Dictionary mapping cluster indices to their intents.

        Returns:
            np.ndarray: A normalized transition matrix representing probabilities of transitions between clusters.
        """
        # Reverse the mapping to get cluster by intent
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
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

        return transition_matrix

    def plot_transition_matrix(transition_matrix: np.ndarray, 
                               intent_by_cluster: Dict[str, str], 
                               font_size: int = 4, 
                               title: str = 'Transition Matrix', 
                               dir_path: str = "output") -> None:
        """
        Plots the transition matrix as a heatmap and saves it as a PNG file.

        Args:
            transition_matrix (np.ndarray): The transition matrix to be plotted.
            intent_by_cluster (Dict[str, str]): Dictionary mapping cluster indices to their intents.
            font_size (int, optional): Font size for the matrix values. Defaults to 4.
            title (str, optional): Title of the plot. Defaults to 'Transition Matrix'.
            dir_path (str, optional): Directory path where the plot will be saved. Defaults to "output".
        """
        fig, ax = plt.subplots(figsize=(50, 30))
        cax = ax.matshow(transition_matrix, cmap='magma_r')

        plt.title(title, pad=20)
        fig.colorbar(cax)
        ax.set_xticks(np.arange(len(intent_by_cluster)))
        ax.set_yticks(np.arange(len(intent_by_cluster)))
        ax.set_xticklabels(intent_by_cluster.values(), rotation=90)
        ax.set_yticklabels(intent_by_cluster.values())

        for (i, j), val in np.ndenumerate(transition_matrix):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=font_size)

        plt.xlabel('To Intent')
        plt.ylabel('From Intent')
        os.makedirs(dir_path, exist_ok=True)

        file_path = os.path.join(dir_path, title.replace(' ', '_') + '.png')
        plt.savefig(file_path, format='png', dpi=300)
        plt.show()
        plt.close()

    def plot_scaled_probability_distribution(transition_matrix: np.ndarray) -> np.ndarray:
        """
        Plots the probability distribution of non-zero transition probabilities on a log scale.

        Args:
            transition_matrix (np.ndarray): The transition matrix containing probabilities.

        Returns:
            np.ndarray: An array of thresholds at the 25th, 50th, and 75th percentiles of the scaled probabilities.
        """
        # Flatten the transition matrix and remove zero entries
        scaled_probabilities = transition_matrix.flatten()
        scaled_probabilities = scaled_probabilities[scaled_probabilities > 0]

        # Calculate complementary percentiles
        thresholds = np.percentile(scaled_probabilities, [25, 50, 75])

        # Define bins in log scale
        bins = np.logspace(np.log10(scaled_probabilities.min()), np.log10(scaled_probabilities.max()), 30)

        plt.figure(figsize=(20, 15))
        plt.hist(scaled_probabilities, bins=bins, alpha=0.75, color='blue', edgecolor='black')
        plt.xscale('log')  # Set x-axis to log scale

        # Add percentile lines
        for threshold, color, label in zip(thresholds, ['red', 'orange', 'green'], 
                                           ['25th percentile', '50th percentile', '75th percentile']):
            plt.axvline(threshold, color=color, linestyle='dashed', linewidth=2, label=f'{label}: {threshold:.2f}')

        plt.title('Probability Distribution of Transitions (Log Scale)')
        plt.xlabel('Scaled Transition Probability (Log Scale)')
        plt.ylabel('Frequency')
        plt.legend()

        # Set custom x-ticks
        x_ticks = np.logspace(np.log10(scaled_probabilities.min()), np.log10(scaled_probabilities.max()), 10)
        plt.xticks(x_ticks, labels=[f'{tick:.2f}' for tick in x_ticks], rotation=90)

        # Add grid lines
        plt.grid(True, which="both", ls="--", linewidth=0.5)

        plt.show()

        return thresholds
