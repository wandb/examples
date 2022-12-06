# import packages
import wandb
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans

# initialize wandb run
run = wandb.init()

# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target
names = iris.target_names


def get_label_ids(classes):
    return np.array([names[aclass] for aclass in classes])


labels = get_label_ids(y)

kmeans = KMeans(n_clusters=4, random_state=1)
cluster_labels = kmeans.fit_predict(X)

# Plot panels to W&B
wandb.sklearn.plot_clusterer(kmeans, X, cluster_labels, labels, "KMeans")

# Finish the W&B Process
run.finish()
