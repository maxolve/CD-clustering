# examples/run_zoo_example.py

from ucimlrepo import fetch_ucirepo
from cdclustering.clustering import run_CDClustering
from cdclustering.evaluation import calculate_modularity

# Fetch the dataset
zoo = fetch_ucirepo(id=111)

# Data (as pandas dataframes)
X = zoo.data.features
y = zoo.data.targets # optional ground truth labels for ARI calculation, not necessary for clustering.

# Metadata
print(zoo.metadata)

# Variable information
print(zoo.variables)

# Number of clusters (assuming we want to find 7 clusters as per the Zoo dataset)
num_clusters = 7

# Run CDClustering
labels = run_CDClustering(X, num_clusters)

# Calculate modularity
modularity_score = calculate_modularity(X, labels)
print(f"Modularity Score: {modularity_score}")
