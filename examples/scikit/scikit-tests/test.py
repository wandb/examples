from sklearn.ensemble import RandomForestClassifier
import wandb

# initialize wandb run
wandb.init(project="test")

# Train model, get predictions
model = RandomForestClassifier()
x_train = [[1, 2]] * 20
x_train += [[3, 4]] * 20

y_train = [0] * 20
y_train += [1] * 20

model.fit(x_train, y_train)
x_test = [[4, 5], [5, 6]]
y_test = [0, 1]
y_probas = model.predict_proba(x_test)
y_pred = model.predict(x_test)

# Visualize model performance
wandb.sklearn.plot_classifier(
    model,
    x_train,
    x_test,
    y_train,
    y_test,
    y_pred,
    y_probas,
    None,
    False,
    "NaiveBayes",
    None,
)
