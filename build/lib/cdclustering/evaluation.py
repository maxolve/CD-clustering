# cdclustering/evaluation.py

import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import pdist, squareform

class ClusteringModularity:
    def __init__(self, dataframe):
        self.df = dataframe
        self.encoded_df = self.encode_categorical_dataframe(dataframe)

    def encode_categorical_dataframe(self, df):
        encoders = {col: LabelEncoder().fit(df[col]) for col in df}
        encoded_df = df.copy()
        for col, encoder in encoders.items():
            encoded_df[col] = encoder.transform(df[col])
        return encoded_df

    def hamming_distance(self, x, y):
        return np.sum(x != y)

    def compute_pairwise_hamming_distances(self, data):
        n = len(data)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.hamming_distance(data.iloc[i], data.iloc[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances

    def compute_cdf(self, distances):
        hist, bins = np.histogram(distances.ravel(), bins=range(int(np.max(distances)) + 2), density=True)
        cdf = np.cumsum(hist * np.diff(bins))
        return cdf, bins

    def estimate_distance_threshold(self, cdf, bins, k):
        for i in range(len(cdf)):
            if cdf[i] >= 1/k:
                return bins[i]
        return bins[-1]

    def create_adjacency_matrix(self, distances, threshold):
        adjacency_matrix = (distances <= threshold).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)
        return adjacency_matrix

    def calculate_modularity(self, adjacency_matrix, cluster_labels):
        G = nx.from_numpy_array(adjacency_matrix)
        communities = {}
        for node, label in enumerate(cluster_labels):
            if label not in communities:
                communities[label] = []
            communities[label].append(node)
        community_list = list(communities.values())
        modularity = nx.algorithms.community.modularity(G, community_list)
        return modularity

def modularity_with_threshold(data, labels, k):
    clustering_modularity = ClusteringModularity(data)
    distances = clustering_modularity.compute_pairwise_hamming_distances(clustering_modularity.encoded_df)
    cdf, bins = clustering_modularity.compute_cdf(distances)
    threshold = clustering_modularity.estimate_distance_threshold(cdf, bins, k)
    if threshold == 0:
        threshold = 1
    adjacency_matrix = clustering_modularity.create_adjacency_matrix(distances, threshold)
    cluster_labels = labels
    modularity_score = clustering_modularity.calculate_modularity(adjacency_matrix, cluster_labels)
    return modularity_score

def create_adjacency_matrix_using_hamming(df):
    distances = pdist(df.values, metric='jaccard')
    distance_matrix = squareform(distances)
    similarity_matrix = 1 - distance_matrix
    return similarity_matrix

def calculate_modularity(dataframe, cluster_labels):
    adjacency_matrix = create_adjacency_matrix_using_hamming(dataframe)
    G = nx.from_numpy_array(adjacency_matrix)
    communities = {}
    for node, label in enumerate(cluster_labels):
        if label not in communities:
            communities[label] = []
        communities[label].append(node)
    community_list = list(communities.values())
    modularity = nx.algorithms.community.modularity(G, community_list)
    return modularity
