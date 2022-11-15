import glob
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from itertools import cycle, islice
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.linear_model as lm
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
import wandb

# Load data
pulsar = pd.read_csv('pulsar_stars.csv')

# Get numeric labels for each of the string labels, to make them compatible with our model
labels_to_class = {'Pulsar': 0, 'Not a Pulsar': 1}
def get_class_ids(labels):
    return np.array([labels_to_class[alabel] for alabel in labels])
def get_named_labels(labels, numeric_labels):
        return np.array([labels[num_label] for num_label in numeric_labels])

# Remove target variables label (and class)
features = list(set(pulsar.columns) - {'target_class'})
X = pulsar[features]
y = pulsar['target_class']
labels = ['Pulsar', 'Not a Pulsar']
X = X[:50000]
y = y[:50000]

scaler = MinMaxScaler(feature_range=(0,1))
features_scaled = scaler.fit_transform(X)
# Split into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

# Clustering - predict particle clusters without labels
# KMeans
kmeans = KMeans(n_clusters=2, random_state=1)
cluster_labels = kmeans.fit_predict(X_train)
wandb.init(project="visualize-sklearn", name='KMeans', reinit=True)
label_names = get_named_labels(labels, cluster_labels)
wandb.sklearn.plot_clusterer(kmeans, X_train, cluster_labels, labels, 'KMeans')
# wandb.sklearn.plot_elbow_curve(model, X_train)

# Classification - predict pulsar
# Train a model, get predictions
log = lm.LogisticRegression(random_state=4)
knn = KNeighborsClassifier(n_neighbors=2)
dtree = DecisionTreeClassifier(random_state=4)
rtree = RandomForestClassifier(n_estimators=100, random_state=4)
svm = SVC(random_state=4, probability=True)
nb = GaussianNB()
gbc = GradientBoostingClassifier()
adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=0.01, random_state=42,
                             base_estimator=DecisionTreeClassifier(max_depth=8,
                             min_samples_leaf=10, random_state=42))

def model_algorithm(clf, X_train, y_train, X_test, y_test, name, labels, features):
    clf.fit(X_train, y_train)
    y_probas = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    run = wandb.init(project="visualize-sklearn", name=name, reinit=True)
    # wandb.sklearn.plot_roc(y_test, y_probas, labels, reinit = True)
    wandb.sklearn.plot_classifier(clf, X_train, X_test, y_train,
                    y_test, y_pred, y_probas, labels, True, name, features)
    run.finish()

model_algorithm(log, X_train, y_train, X_test, y_test, 'LogisticRegression', labels, features)
model_algorithm(knn, X_train, y_train, X_test, y_test, 'KNearestNeighbor', labels, features)
model_algorithm(dtree, X_train, y_train, X_test, y_test, 'DecisionTree', labels, features)
model_algorithm(rtree, X_train, y_train, X_test, y_test, 'RandomForest', labels, features)
model_algorithm(svm, X_train, y_train, X_test, y_test, 'SVM', labels, features)
model_algorithm(nb, X_train, y_train, X_test, y_test, 'NaiveBayes', labels, features)
model_algorithm(adaboost, X_train, y_train, X_test, y_test, 'AdaBoost', labels, features)
model_algorithm(gbc, X_train, y_train, X_test, y_test, 'GradientBoosting', labels, features)

# Regression - TrackP - particle momentum
features = list(set(pulsar.columns) - {' Mean of the integrated profile'})
X = pulsar[features]
y = pulsar[' Mean of the integrated profile']
X = X[:10000]
y = y[:10000]

# Split into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.90, test_size=0.10)

# Train a model, get predictions
reg = Ridge()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Visualize model performance
wandb.init(name='Ridge', reinit=True)
wandb.sklearn.plot_regressor(reg, X_train, X_test,
                              y_train, y_test, 'Ridge')
