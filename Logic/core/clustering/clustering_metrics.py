import numpy as np

from typing import List
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


class ClusteringMetrics:

    def __init__(self):
        pass

    def silhouette_score(self, embeddings: List, cluster_labels: List) -> float:
        """
        Calculate the average silhouette score for the given cluster assignments.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points.
        cluster_labels: List
            A list of cluster assignments for each data point.

        Returns
        --------
        float
            The average silhouette score, ranging from -1 to 1, where a higher value indicates better clustering.
        """
        embeddings = np.array(embeddings)
        cluster_labels = np.array(cluster_labels)
        return silhouette_score(embeddings, cluster_labels)

    def purity_score(self, true_labels: List, cluster_labels: List) -> float:
        """
        Calculate the purity score for the given cluster assignments and ground truth labels.

        Parameters
        -----------
        true_labels: List
            A list of ground truth labels for each data point (Genres).
        cluster_labels: List
            A list of cluster assignments for each data point.

        Returns
        --------
        float
            The purity score, ranging from 0 to 1, where a higher value indicates better clustering.
        """
        # single_labels = [np.bincount(cluster_labels).argmax() for labels in true_labels]
        # if not single_labels:
        #    return 0.0
        # matrix = confusion_matrix(single_labels, cluster_labels)

        matrix = confusion_matrix(true_labels, cluster_labels)
        maxes = np.max(matrix, axis=0)
        return np.sum(maxes) / np.sum(matrix)

    def adjusted_rand_score(self, true_labels: List, cluster_labels: List) -> float:
        """
        Calculate the adjusted Rand index for the given cluster assignments and ground truth labels.

        Parameters
        -----------
        true_labels: List
            A list of ground truth labels for each data point (Genres).
        cluster_labels: List
            A list of cluster assignments for each data point.

        Returns
        --------
        float
            The adjusted Rand index, ranging from -1 to 1, where a higher value indicates better clustering.
        """
        # unique_clusters = np.unique(cluster_labels)
        # cluster_true_labels = []
        # for cluster in unique_clusters:
        #    true_labels_in_cluster = [true_labels[i] for i, label in enumerate(cluster_labels) if label == cluster]
        #    cluster_true_labels.append(
        #        max(set([tuple(label) for label in true_labels_in_cluster]), key=true_labels_in_cluster.count))

        # return adjusted_rand_score(true_labels, cluster_true_labels)

        return adjusted_rand_score(true_labels, cluster_labels)
