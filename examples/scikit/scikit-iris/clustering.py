import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets, cluster
from sklearn.datasets import load_iris
import numpy as np
import wandb

# initialize wandb run
wandb.init()

iris = datasets.load_iris()
X = iris.data
y = iris.target

kmeans = KMeans(n_clusters=4, random_state=1)
cluster_labels = kmeans.fit_predict(X)

wandb.sklearn.plot_elbow_curve(kmeans, X)
wandb.sklearn.plot_silhouette(kmeans, X, cluster_labels)
