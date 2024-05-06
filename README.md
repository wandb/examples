<p align="center">
  <img src="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg#gh-light-mode-only" width="600" alt="Weights & Biases"/>
  <img src="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-dark.svg#gh-dark-mode-only" width="600" alt="Weights & Biases"/>
</p>

Use W&B to build better models faster. Track and visualize all the pieces of your machine learning pipeline, from datasets to production machine learning models. Get started with W&B today, [sign up for a free account!](https://wandb.com?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme)

&nbsp;

<p align='center'>
<a target="_blank" href="https://docs.wandb.ai/guides/track?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_dark_background/experiments-dark.svg" width="13.5%">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_light/experiments-light.svg" width="13.5%">
  <img alt="Weights and Biases Experiments" src="">
</picture>
</a>
<a target="_blank" href="https://docs.wandb.ai/guides/reports?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_dark_background/report-dark.svg" width="13.5%">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_light/report-light.svg" width="13.5%">
  <img alt="Weights and Biases Reports" src="">
</picture>
</a>
<a target="_blank" href="https://docs.wandb.ai/guides/artifacts?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_dark_background/artifacts-dark.svg" width="13.5%">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_light/artifacts-light.svg" width="13.5%">
  <img alt="Weights and Biases Artifacts" src="">
</picture>
</a>
<a target="_blank" href="https://docs.wandb.ai/guides/data-vis?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_dark_background/tables-dark.svg" width="13.5%">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_light/tables-light.svg" width="13.5%">
  <img alt="Weights and Biases Tables" src="">
</picture>
</a>
<a target="_blank" href="https://docs.wandb.ai/guides/sweeps?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_dark_background/sweeps-dark.svg" width="13.5%">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_light/sweeps-light.svg" width="13.5%">
  <img alt="Weights and Biases Sweeps" src="">
</picture>
</a>
<a target="_blank" href="https://docs.wandb.ai/guides/models?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_dark_background/models-dark.svg" width="13.5%">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_light/models-light.svg" width="13.5%">
  <img alt="Weights and Biases Model Management" src="">
</picture>
</a>
<a target="_blank" href="https://docs.wandb.ai/guides/launch?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_dark_background/launch-dark.svg" width="13.5%">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/Product_Icons_light/launch-light.svg" width="13.5%">
  <img alt="Weights and Biases Launch" src="">
</picture>
</a>
</p>

&nbsp;

# 🚀 Getting Started

### Never lose your progress again. 
Save everything you need to compare and reproduce models — architecture, hyperparameters, weights, model predictions, GPU usage, git commits, and even datasets — in 5 minutes. W&B is free for personal use and academic projects, and it's easy to get started.

**Check out our libraries of [example scripts](https://github.com/wandb/examples/tree/master/examples) and [example colabs](https://github.com/wandb/examples/tree/master/colabs)**
or read on for code snippets and more!

If you have any questions, please don't hesitate to ask in our [Discourse forum](http://wandb.me/and-you).

# 🤝 Simple integration with any framework
Install `wandb` library and login:
```
pip install wandb
wandb login
```
Flexible integration for any Python script:
```python
import wandb

# 1. Start a W&B run
wandb.init(project='gpt3')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01

# Model training code here ...

# 3. Log metrics over time to visualize performance
for i in range (10):
    wandb.log({"loss": loss})
```

### [Try in a colab →](http://wandb.me/intro-colab)

If you have any questions, please don't hesitate to ask in our [Discourse forum](http://wandb.me/and-you).

![](https://i.imgur.com/TU34QFZ.png)

**[Explore a W&B dashboard](https://www.youtube.com/watch?v=gnD8BFuyVUA)**

# 📈 Track model and data pipeline hyperparameters
Set `wandb.config` once at the beginning of your script to save your hyperparameters, input settings (like dataset name or model type), and any other independent variables for your experiments. This is useful for analyzing your experiments and reproducing your work in the future. Setting configs also allows you to [visualize](https://docs.wandb.com/sweeps/visualize-sweep-results) the relationships between features of your model architecture or data pipeline and the model performance (as seen in the screenshot above).

```python
wandb.init()
wandb.config.epochs = 4
wandb.config.batch_size = 32
wandb.config.learning_rate = 0.001
wandb.config.architecture = "resnet"
```

- **[See how to set configs in a colab →](http://wandb.me/config-colab)**
- [Docs](https://docs.wandb.com/library/config)

# 🏗 Use your favorite framework

Use your favorite framework with W&B. W&B integrations make it fast and easy to set up experiment tracking and data versioning inside existing projects. For more information on how to integrate W&B with the framework of your choice, see the [Integrations chapter](https://docs.wandb.ai/guides/integrations) in the W&B Developer Guide.

<!-- <p align='center'>
<img src="./docs/README_images/integrations.png" width="100%" />
</p> -->

<details>
<summary>🔥 PyTorch</summary>

Call `.watch` and pass in your PyTorch model to automatically log gradients and store the network topology. Next, use `.log` to track other metrics. The following example demonstrates an example of how to do this:

```python
import wandb

# 1. Start a new run
run = wandb.init(project="gpt4")

# 2. Save model inputs and hyperparameters
config = run.config
config.dropout = 0.01

# 3. Log gradients and model parameters
run.watch(model)
for batch_idx, (data, target) in enumerate(train_loader):
    ...
    if batch_idx % args.log_interval == 0:
        # 4. Log metrics to visualize performance
        run.log({"loss": loss})
```

- Run an example [Google Colab Notebook](http://wandb.me/pytorch-colab).
- Read the [Developer Guide](https://docs.wandb.com/guides/integrations/pytorch?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=integrations) for technical details on how to integrate PyTorch with W&B.
- Explore [W&B Reports](https://app.wandb.ai/wandb/getting-started/reports/Pytorch--VmlldzoyMTEwNzM?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=integrations).

</details>
<details>
<summary>🌊 TensorFlow/Keras</summary>
Use W&B Callbacks to automatically save metrics to W&B when you call `model.fit` during training.

The following code example demonstrates how your script might look like when you integrate W&B with Keras:

```python
# This script needs these libraries to be installed:
#   tensorflow, numpy

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import random
import numpy as np
import tensorflow as tf


# Start a run, tracking hyperparameters
run = wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    # track hyperparameters and run metadata with wandb.config
    config={
        "layer_1": 512,
        "activation_1": "relu",
        "dropout": random.uniform(0.01, 0.80),
        "layer_2": 10,
        "activation_2": "softmax",
        "optimizer": "sgd",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 8,
        "batch_size": 256,
    },
)

# [optional] use wandb.config as your config
config = run.config

# get the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, y_train = x_train[::5], y_train[::5]
x_test, y_test = x_test[::20], y_test[::20]
labels = [str(digit) for digit in range(np.max(y_train) + 1)]

# build a model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(config.layer_1, activation=config.activation_1),
        tf.keras.layers.Dropout(config.dropout),
        tf.keras.layers.Dense(config.layer_2, activation=config.activation_2),
    ]
)

# compile the model
model.compile(optimizer=config.optimizer, loss=config.loss, metrics=[config.metric])

# WandbMetricsLogger will log train and validation metrics to wandb
# WandbModelCheckpoint will upload model checkpoints to wandb
history = model.fit(
    x=x_train,
    y=y_train,
    epochs=config.epoch,
    batch_size=config.batch_size,
    validation_data=(x_test, y_test),
    callbacks=[
        WandbMetricsLogger(log_freq=5),
        WandbModelCheckpoint("models"),
    ],
)

# [optional] finish the wandb run, necessary in notebooks
run.finish()
```

Get started integrating your Keras model with W&B today:

- Run an example [Google Colab Notebook](https://wandb.me/intro-keras?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=integrations)
- Read the [Developer Guide](https://docs.wandb.com/guides/integrations/keras?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=integrations) for technical details on how to integrate Keras with W&B.
- Explore [W&B Reports](https://app.wandb.ai/wandb/getting-started/reports/Keras--VmlldzoyMTEwNjQ?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=integrations).

</details>
<details>
<summary>🤗 Huggingface Transformers</summary>

Pass `wandb` to the `report_to` argument when you run a script using a HuggingFace Trainer. W&B will automatically log losses,
evaluation metrics, model topology, and gradients.

**Note**: The environment you run your script in must have `wandb` installed.

The following example demonstrates how to integrate W&B with Hugging Face:

```python
# This script needs these libraries to be installed:
#   numpy, transformers, datasets

import wandb

import os
import numpy as np
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}


# download prepare the data
dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(300))

small_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
small_eval_dataset = small_train_dataset.map(tokenize_function, batched=True)

# download the model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=5
)

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = "my-awesome-project"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "true"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"

# pass "wandb" to the `report_to` parameter to turn on wandb logging
training_args = TrainingArguments(
    output_dir="models",
    report_to="wandb",
    logging_steps=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=20,
    max_steps=100,
    save_steps=100,
)

# define the trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
```

- Run an example [Google Colab Notebook](http://wandb.me/hf?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=integrations).
- Read the [Developer Guide](https://docs.wandb.com/guides/integrations/huggingface?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=integrations) for technical details on how to integrate Hugging Face with W&B.
</details>

<details>
<summary>⚡️ PyTorch Lightning</summary>

Build scalable, structured, high-performance PyTorch models with Lightning and log them with W&B.

```python
# This script needs these libraries to be installed:
#   torch, torchvision, pytorch_lightning

import wandb

import os
from torch import optim, nn, utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, lr=1e-3, inp_size=28, optimizer="Adam"):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inp_size * inp_size, 64), nn.ReLU(), nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, inp_size * inp_size)
        )
        self.lr = lr

        # save hyperparameters to self.hparamsm auto-logged by wandb
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)

        # log metrics to wandb
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(lr=1e-3, inp_size=28)

# setup data
batch_size = 32
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset, shuffle=True)

# initialise the wandb logger and name your wandb project
wandb_logger = WandbLogger(project="my-awesome-project")

# add your batch size to the wandb config
wandb_logger.experiment.config["batch_size"] = batch_size

# pass wandb_logger to the Trainer
trainer = pl.Trainer(limit_train_batches=750, max_epochs=5, logger=wandb_logger)

# train the model
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
```

- Run an example [Google Colab Notebook](http://wandb.me/lightning?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=integrations).
- Read the [Developer Guide](https://docs.wandb.ai/guides/integrations/lightning?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=integrations) for technical details on how to integrate PyTorch Lightning with W&B.
</details>
<details>
<summary>💨 XGBoost</summary>
Use W&B Callbacks to automatically save metrics to W&B when you call `model.fit` during training.

The following code example demonstrates how your script might look like when you integrate W&B with XGBoost:

```python
# This script needs these libraries to be installed:
#   numpy, xgboost

import wandb
from wandb.xgboost import WandbCallback

import numpy as np
import xgboost as xgb


# setup parameters for xgboost
param = {
    "objective": "multi:softmax",
    "eta": 0.1,
    "max_depth": 6,
    "nthread": 4,
    "num_class": 6,
}

# start a new wandb run to track this script
run = wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    # track hyperparameters and run metadata
    config=param,
)

# download data from wandb Artifacts and prep data
run.use_artifact("wandb/intro/dermatology_data:v0", type="dataset").download(".")
data = np.loadtxt(
    "./dermatology.data",
    delimiter=",",
    converters={33: lambda x: int(x == "?"), 34: lambda x: int(x) - 1},
)
sz = data.shape

train = data[: int(sz[0] * 0.7), :]
test = data[int(sz[0] * 0.7) :, :]

train_X = train[:, :33]
train_Y = train[:, 34]

test_X = test[:, :33]
test_Y = test[:, 34]

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
watchlist = [(xg_train, "train"), (xg_test, "test")]

# add another config to the wandb run
num_round = 5
run.config["num_round"] = 5
run.config["data_shape"] = sz

# pass WandbCallback to the booster to log its configs and metrics
bst = xgb.train(
    param, xg_train, num_round, evals=watchlist, callbacks=[WandbCallback()]
)

# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]

# log your test metric to wandb
run.summary["Error Rate"] = error_rate

# [optional] finish the wandb run, necessary in notebooks
run.finish()
```

- Run an example [Google Colab Notebook](https://wandb.me/xgboost?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=integrations).
- Read the [Developer Guide](https://docs.wandb.ai/guides/integrations/xgboost?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=integrations) for technical details on how to integrate XGBoost with W&B.
</details>
<details>
<summary>🧮 Sci-Kit Learn</summary>
Use wandb to visualize and compare your scikit-learn models' performance:

```python
# This script needs these libraries to be installed:
#   numpy, sklearn

import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc

import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# load and process data
wbcd = datasets.load_breast_cancer()
feature_names = wbcd.feature_names
labels = wbcd.target_names

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    wbcd.data, wbcd.target, test_size=test_size
)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
model_params = model.get_params()

# get predictions
y_pred = model.predict(X_test)
y_probas = model.predict_proba(X_test)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# start a new wandb run and add your model hyperparameters
run = wandb.init(project="my-awesome-project", config=model_params)

# Add additional configs to wandb
run.config.update(
    {
        "test_size": test_size,
        "train_len": len(X_train),
        "test_len": len(X_test),
    }
)

# log additional visualisations to wandb
plot_class_proportions(y_train, y_test, labels)
plot_learning_curve(model, X_train, y_train)
plot_roc(y_test, y_probas, labels)
plot_precision_recall(y_test, y_probas, labels)
plot_feature_importances(model)

# [optional] finish the wandb run, necessary in notebooks
run.finish()
```

- Run an example [Google Colab Notebook](https://wandb.me/scikit-colab?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=integrations).
- Read the [Developer Guide](https://docs.wandb.ai/guides/integrations/scikit?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=integrations) for technical details on how to integrate Scikit-Learn with W&B.
</details>

&nbsp;
# 🧹 Optimize hyperparameters with Sweeps
Use Weights & Biases Sweeps to automate hyperparameter optimization and explore the space of possible models.

### [Try Sweeps in PyTorch in a Colab →](http://wandb.me/sweeps-colab)
### [Try Sweeps in TensorFlow in a Colab →](http://wandb.me/tf-sweeps-colab)

### Benefits of using W&B Sweeps 
- **Quick to setup:** With just a few lines of code you can run W&B sweeps.
- **Transparent:** We cite all the algorithms we're using, and our code is [open source](https://github.com/wandb/client/blob/master/wandb/sdk/wandb_sweep.py).
- **Powerful:** Our sweeps are completely customizable and configurable. You can launch a sweep across dozens of machines, and it's just as easy as starting a sweep on your laptop.

### [Get started in 5 mins →](https://docs.wandb.com/sweeps/quickstart)

<img src="https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-LyfPCyvV8By5YBltxfh%2F-LyfQsxswLC-6WKGgfGj%2Fcentral%20sweep%20server%203.png?alt=media&token=c81e4fe7-7ee4-48ea-a4cd-7b28113c6088" width="400" alt="Weights & Biases" />

### Common use cases
- **Explore:** Efficiently sample the space of hyperparameter combinations to discover promising regions and build an intuition about your model.
- **Optimize:**  Use sweeps to find a set of hyperparameters with optimal performance.
- **K-fold cross validation:** [Here's a brief code example](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation) of _k_-fold cross validation with W&B Sweeps.

### Visualize Sweeps results
The hyperparameter importance plot surfaces which hyperparameters were the best predictors of, and highly correlated to desirable values for your metrics.

<img src="https://paper-attachments.dropbox.com/s_194708415DEC35F74A7691FF6810D3B14703D1EFE1672ED29000BA98171242A5_1578695757573_image.png" width="720" alt="Weights & Biases" />

Parallel coordinates plots map hyperparameter values to model metrics. They're useful for honing in on combinations of hyperparameters that led to the best model performance.

<img src="https://i.imgur.com/THYXBN0.png" width="720" alt="Weights & Biases" />

# 📜 Share insights with Reports
Reports let you [organize visualizations, describe your findings, and share updates with collaborators](http://wandb.me/reports-guide).

### Common use cases
- **Notes:** Add a graph with a quick note to yourself.
- **Collaboration:** Share findings with your colleagues.
- **Work log:** Track what you've tried and plan next steps.

**Explore reports in [The Gallery →](https://wandb.ai/gallery) | Read the [Docs](https://docs.wandb.com/reports)**

Once you have experiments in W&B, you can visualize and document results in Reports with just a few clicks. Here's a quick [demo video](http://wandb.me/short-reports).

![](https://i.imgur.com/dn0Dyd8.png)

# 🏺 Version control datasets and models with Artifacts
Git and GitHub make code version control easy,
but they're not optimized for tracking the other parts of the ML pipeline:
datasets, models, and other large binary files.

W&B's Artifacts are.
With just a few extra lines of code,
you can start tracking you and your team's outputs,
all directly linked to run.

### Try Artifacts in a [Colab](http://wandb.me/artifacts-colab) with a [video tutorial](http://wandb.me/artifacts-video)

![](https://i.imgur.com/zvBWhGx.png)

### Common use cases
- **Pipeline Management:** Track and visualize the inputs and outputs of your runs as a graph
- **Don't Repeat Yourself™:** Prevent the duplication of compute effort
- **Sharing Data in Teams:** Collaborate on models and datasets without all the headaches

![](https://i.imgur.com/w92cYQm.png)

**Learn about Artifacts [here →](https://www.wandb.com/articles/announcing-artifacts) | Read the [Docs](https://docs.wandb.com/artifacts)**



# Visualize and Query data with Tables

Group, sort, filter, generate calculated columns, and create charts from tabular data.

Spend more time deriving insights, and less time building charts manually.

```
# log my table

wandb.log({"table": my_dataframe})
```

![](https://i.imgur.com/Fg9xR6M.gif)

### Try Tables in a [Colab](http://wandb.me/tables-quickstart) or these [examples](https://github.com/wandb/examples/tree/master/colabs/tables)

**Explore Tables [here →](https://wandb.ai/site/tables) | Read the [Docs](https://docs.wandb.ai/guides/data-vis)**
