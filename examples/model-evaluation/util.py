'''
util.py

Author: Tim Sweeney
'''

import wandb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split

def generate_raw_data(train_size=60000):
    eval_size = int(train_size / 6)
    (x_train, y_train), (x_eval, y_eval) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_eval = x_eval.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_eval = np.expand_dims(x_eval, -1)
    print("::util.generate_raw_data:: Generated {} rows of training data and {} rows of eval data.".format(train_size, eval_size))
    return (x_train[:train_size], y_train[:train_size]), (x_eval[:eval_size], y_eval[:eval_size])

def publish_dataset_to_wb(x_train, y_train, x_eval, y_eval, model_use_case_id="mnist"):
    name = "{}_ds".format(model_use_case_id)
    artifact = wandb.Artifact(name, "dataset")
    
    train_table = wandb.Table(data=[], columns=[])
    train_table.add_column("x_train", x_train)
    train_table.add_column("y_train", y_train)
    train_table.add_computed_columns(lambda ndx, row: {
        "img": wandb.Image(row["x_train"])
    })
    
    eval_table = wandb.Table(data=[], columns=[])
    eval_table.add_column("x_eval", x_eval)
    eval_table.add_column("y_eval", y_eval)
    eval_table.add_computed_columns(lambda ndx, row: {
        "img": wandb.Image(row["x_eval"])
    })
    
    artifact.add(train_table, "train_table")
    artifact.add(eval_table, "eval_table")
    artifact.save()
    print("::util.publish_dataset_to_wb:: Published data to Artifact {}".format(name))

def download_training_dataset_from_wb(model_use_case_id="mnist", version="latest"):
    name = "{}:{}".format("{}_ds".format(model_use_case_id), version)
    artifact = wandb.run.use_artifact(name)
    print("::util.download_training_dataset_from_wb:: Downlaoding Artifact {}".format(artifact.name))
    train_table = artifact.get("train_table")
    x_train = train_table.get_column("x_train", convert_to="numpy")
    y_train = train_table.get_column("y_train", convert_to="numpy")
    return x_train, y_train


def build_and_train_model(x_train, y_train, config):
    print("::util.build_and_train_model:: Building model with config {}".format(config))
    num_classes=10
    input_shape=(28, 28, 1)
    loss="categorical_crossentropy"
    optimizer=config.optimizer
    metrics=["accuracy"]
    batch_size=config.batch_size
    epochs=config.epochs
    validation_split=config.validation_split
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)
    model.fit(x_t, y_t, batch_size=batch_size, epochs=epochs, validation_data=(x_v, y_v), callbacks=[WandbCallback(
        log_weights=True,
        log_evaluation=True
    )])
    return model

def publish_model_candidate_to_wb(model, model_use_case_id):
    name = "{}_model_candidates".format(model_use_case_id)
    print("::util.publish_model_candidate_to_wb:: Publishing model version to Artifact {}".format(name))
    artifact = wandb.Artifact(name, "model")
    path = "{}.h5".format(np.random.randint(1e5))
    model.save(path)
    artifact.add_file(path, "model.h5")
    artifact.save()

def download_eval_dataset_from_wb(model_use_case_id="mnist", version="latest"):
    artifact = wandb.run.use_artifact("{}:{}".format("{}_ds".format(model_use_case_id), version))
    print("::util.download_eval_dataset_from_wb:: Downloading latest validation dataset {}".format(artifact.name))
    eval_table = artifact.get("eval_table")
    x_eval = eval_table.get_column("x_eval", convert_to="numpy")
    y_eval = eval_table.get_column("y_eval", convert_to="numpy")
    return x_eval, y_eval, artifact


def get_new_model_candidates_from_wb(project, model_use_case_id, metric_key):
    model_candidates = _get_model_candidates_from_wb(project, model_use_case_id)
    unevaluated_candidates = []
    min_loss = float("inf")
    for candidate in model_candidates:
        if (metric_key not in candidate.metadata):
            unevaluated_candidates.append(candidate)
        else:
            min_loss = min(min_loss, candidate.metadata[metric_key])

    print("::util.filter_model_candidates:: Total unevaluated models on {}: {}/{}".format(metric_key, len(unevaluated_candidates), len(model_candidates)))
    return unevaluated_candidates


def _get_model_candidates_from_wb(project, model_use_case_id):
    api = wandb.Api({"project": project})
    versions = api.artifact_versions("model", "{}_model_candidates".format(model_use_case_id))
    return versions

def evaluate_model(model_artifact, x_eval, y_eval):
    art = wandb.run.use_artifact(model_artifact)
    model = keras.models.load_model(art.get_path("model.h5").download())
    y_eval = keras.utils.to_categorical(y_eval, 10)
    (loss, _) = model.evaluate(x_eval, y_eval)
    return loss


def save_metric_to_model_in_wb(model, metric, score):    
    print("::util.save_metric_to_model_in_wb:: Saving score of {} on metric {} to {}".format(score, metric, model.name))
    model.metadata[metric] = score
    model.save()
        

def promote_best_model_in_wb(project, model_use_case_id, metric):
    all_candidates = _get_model_candidates_from_wb(project, model_use_case_id)
    best_model = None
    best_loss = float("inf")
    for model in all_candidates:
        if metric in model.metadata and model.metadata[metric] < best_loss:
            best_model = model
            best_loss = model.metadata[metric]
    
    if (best_model is None):
        print("::util.promote_best_model_in_wb:: No valid model found")
    else:
        if ('production' in best_model.aliases):
            print("::util.promote_best_model_in_wb:: Existing production model {} has best score {} on {}".format(best_model.name, best_loss, metric, metric))
        else:
            print("::util.promote_best_model_in_wb:: Promoting model {} with best score {} on {} to production".format(best_model.name, best_loss, metric, metric))
            best_model.aliases.append('production')
            best_model.save()