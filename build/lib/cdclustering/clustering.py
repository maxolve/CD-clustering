# cdclustering/clustering.py

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain  # Ensure this matches the import in your notebook

class CDClustering:
    def __init__(self):
        pass

    def hamming_distance(self, x, y):
        """Calculate the Hamming distance between two arrays."""
        return np.sum(x != y)

    def compute_pairwise_hamming_distances(self, data):
        """Compute pairwise Hamming distances for all pairs in the dataset."""
        n = len(data)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.hamming_distance(data.iloc[i], data.iloc[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances

    def compute_cdf(self, distances):
        """Compute the cumulative distribution function of the Hamming distances."""
        hist, bins = np.histogram(distances.ravel(), bins=range(int(np.max(distances)) + 2), density=True)
        cdf = np.cumsum(hist * np.diff(bins))
        return cdf, bins

    def estimate_distance_threshold(self, cdf, bins, k):
        """Estimate the Hamming distance threshold R."""
        for i in range(len(cdf)):
            if cdf[i] <= 1/k and (i == len(cdf) - 1 or cdf[i + 1] > 1/k):
                return bins[i]
        return bins[-1]

    def build_graph(self, data, threshold):
        """Build a graph where edges exist between nodes with Hamming distance <= threshold."""
        n = len(data)
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                if self.hamming_distance(data.iloc[i], data.iloc[j]) <= threshold:
                    G.add_edge(i, j)
        return G

    def detect_communities(self, G):
        """Detect communities in the graph using the Louvain method."""
        partition = community_louvain.best_partition(G)
        return partition

    def form_clusters_with_labels(self, data, partition, k):
        """Form the final clusters from the partition and return cluster labels for each data point."""
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)

        sorted_communities = sorted(communities.items(), key=lambda item: len(item[1]), reverse=True)[:k]
        clusters = {i: data.iloc[comm[1]] for i, comm in enumerate(sorted_communities)}

        modes = {}
        for idx, cluster in clusters.items():
            modes[idx] = cluster.mode().iloc[0]

        labels = [None] * len(data)

        for idx, indices in clusters.items():
            for i in indices.index:
                labels[i] = idx

        for idx, row in data.iterrows():
            if labels[idx] is not None:
                continue
            nearest_mode = min(modes.items(), key=lambda mode: self.hamming_distance(row, mode[1]))
            labels[idx] = nearest_mode[0]

        return labels

def run_CDClustering(categorical_data, num_clusters):
    cd_clustering = CDClustering()
    distances = cd_clustering.compute_pairwise_hamming_distances(categorical_data)
    cdf, bins = cd_clustering.compute_cdf(distances)
    k = num_clusters
    R = cd_clustering.estimate_distance_threshold(cdf, bins, k)
    G = cd_clustering.build_graph(categorical_data, R)
    partition = cd_clustering.detect_communities(G)
    labels_graph_cluster = cd_clustering.form_clusters_with_labels(categorical_data, partition, k)
    return labels_graph_cluster
