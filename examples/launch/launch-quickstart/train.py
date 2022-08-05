import wandb
import numpy as np
import tensorflow as tf

# Training hyperparameters
config = {
  "layer_1": 512,
  "activation_1": "relu",
  "dropout": 0.25,
  "layer_2": 10,
  "activation_2": "softmax",
  "optimizer": "sgd",
  "loss": "sparse_categorical_crossentropy",
  "metric": "accuracy",
  "epoch": 10,
  "batch_size": 256,
}

TABLE_COLUMNS = ["image", "label", "prediction"]
TABLE_COLUMNS.extend([f"score_{i}" for i in range(10)])

def build_pred_table(dataset, logits):
    data = [
        [wandb.Image(dataset["x"][idx]), dataset["y"][idx], np.argmax(logits[idx]), *logits[idx]]
        for idx in range(len(logits))
    ]
    table = wandb.Table(data=data, columns=TABLE_COLUMNS)
    return table

def build_model(config):
  """Construct a simple neural network."""
  model = tf.keras.models.Sequential(
    [
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(config.layer_1, activation=config.activation_1),
      tf.keras.layers.Dropout(config.dropout),
      tf.keras.layers.Dense(config.layer_2, activation=config.activation_2),
    ]
  )
  model.compile(optimizer=config.optimizer, loss=config.loss, metrics=[config.metric])
  return model

with wandb.init(config=config) as run:
    
    # Build model and prepare data
    model = build_model(run.config)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    model.fit(
        x=x_train,
        y=y_train,
        epochs=run.config.epoch,
        batch_size=run.config.batch_size,
        validation_data=(x_test[:100], y_test[:100]),
        callbacks=[wandb.keras.WandbCallback(log_model=True, log_evaluation=True)],
    )

    x_test = x_test[100:300]
    y_test = y_test[100:300]
    logits = model.predict(x_test)
    preds = np.argmax(logits, axis=1)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()(y_test, logits).numpy()
    accuracy = tf.keras.metrics.Accuracy()(y_test, preds).numpy()
    run.summary["test_loss"] = loss
    run.summary["test_accuracy"] = accuracy
    run.log({"predictions": build_pred_table(dict(x=x_test, y=y_test), logits)})