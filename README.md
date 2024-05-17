# CD-Clustering

A package for categorical data clustering using the CD-Clustering algorithm.

## Introduction

CD-Clustering is a method for clustering categorical data by leveraging community detection techniques. This approach was introduced by Huu Hiep Nguyen in the paper "Clustering Categorical Data Using Community Detection Techniques" published in *Computational Intelligence and Neuroscience* in 2017.

## Reference

If you use this package in your research, please cite the following paper:

@article{nguyen2017clustering,
  title={Clustering categorical data using community detection techniques},
  author={Nguyen, Huu Hiep},
  journal={Computational intelligence and neuroscience},
  volume={2017},
  year={2017},
  publisher={Hindawi Limited}
}


## Directory Structure
```text
CDClustering/
│
├── cdclustering/
│ ├── __init__.py
│ ├── clustering.py
│ ├── evaluation.py
│ └── utils.py # Note: draft of a utils file with an exemplary csv import function
│
├── data/
│ └── example.csv # Note: empty csv file for optional data imports
│
├── examples/
│ └── run_zoo_example.py
│
├── README.md
├──  requirements.txt
└── setup.py
```


## Installation

You can install the package using `setup.py`, or `requirements.txt`.

### Using setup.py
1. **Clone the repository:**
```bash
git clone https://github.com/maxolve/CDClustering.git
cd CDClustering
```
2. Install the package and dependencies:
```bash
pip install .
```

### Using requirements.txt
Alternatively, you can install the dependencies directly from requirements.txt:
1. **Clone the repository:**
```bash
git clone <repository-url>
cd CDClustering
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```
## Usage
### Running the CDClustering Algorithm with the UCI Zoo Dataset

1. Fetch the dataset and run the example script:

```bash
python examples/run_zoo_example.py
```
This script will:
- Fetch the UCI Zoo dataset using ucimlrepo.
- Run the CDClustering algorithm to cluster the data.
- Calculate and print the modularity score of the clustering.

modularity_score = calculate_modularity(data, labels)
print(f"Modularity Score: {modularity_score}")
```