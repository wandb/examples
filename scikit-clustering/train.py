import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn import datasets, cluster
from sklearn.datasets import load_iris
import numpy as np
import wandb
wandb.init()

iris = datasets.load_iris()
X = iris.data
y = iris.target
names = iris.target_names

def get_label_ids(classes):
    return np.array([names[aclass] for aclass in classes])
labels = get_label_ids(y)

kmeans = KMeans(n_clusters=4, random_state=1)
dbscan = DBSCAN(eps=0.008, min_samples=2)
agg = AgglomerativeClustering()
cluster_labels = kmeans.fit_predict(X)

# wandb.sklearn.plot_elbow_curve(kmeans, X)
wandb.sklearn.plot_clusterer(kmeans, X, cluster_labels, labels, 'KMeans')
