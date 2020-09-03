from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import wandb

# initialize wandb run
wandb.init()

# load data
iris = load_iris()
X = iris.data
y = iris.target
y[y != 0] = 1

# shuffle data
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# create model
model = RandomForestClassifier()
wandb.sklearn.plot_learning_curve(model, X_test, y_test)
