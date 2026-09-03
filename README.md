# W&B examples

This repository contains scripts and notebooks that show how to use Weights &
Biases (W&B) in machine learning workflows. Use these examples to add
experiment tracking, model and dataset versioning, hyperparameter tuning, rich
media logging, and collaborative reports to your own projects.

For product documentation, see the [W&B Models documentation](https://docs.wandb.ai/) and
the [W&B integrations guide](https://docs.wandb.ai/models/integrations).

## Repository layout

| Path | Contents |
| --- | --- |
| [`colabs/`](colabs/) | Google Colab notebooks for W&B features, frameworks, and workflows. |
| [`examples/`](examples/) | Runnable scripts organized by framework or W&B feature. |
| [`wandb-scim/`](wandb-scim/) | SCIM administration examples. |

The examples use different frameworks and dependency sets. Before running an
example, check the README or `requirements.txt` file in that example's
directory.

## Get started

Install the W&B Python SDK and authenticate your machine:

```bash
pip install --upgrade wandb
wandb login
```

Clone this repository:

```bash
git clone https://github.com/wandb/examples.git
cd examples
```

For a local script, install any example-specific dependencies listed in that
directory and run the training file:

```bash
cd examples/pytorch/pytorch-cnn-fashion
python train.py
```

For notebooks, open the matching Colab from [`colabs/README.md`](colabs/README.md)
or run the notebook locally in your preferred environment.

## Track an experiment

Use `wandb.init()` to create a run, `run.config` to store input settings, and
`run.log()` to record metrics during training.

```python
import wandb
import random

# Project that the run is recorded to
project = "my-awesome-project"

# Dictionary with hyperparameters
config = {
    'epochs' : 10,
    'lr' : 0.01
}

with wandb.init(project=project, config=config) as run:
    offset = random.random() / 5
    print(f"lr: {config['lr']}")
    
    # Simulate a training run
    for epoch in range(2, config['epochs']):
        acc = 1 - 2**-config['epochs'] - random.random() / config['epochs'] - offset
        loss = 2**-config['epochs'] + random.random() / config['epochs'] + offset
        print(f"epoch={config['epochs']}, accuracy={acc}, loss={loss}")
        run.log({"accuracy": acc, "loss": loss})
```

Configuration values are for inputs and independent variables, such as learning
rate, batch size, dataset name, and model architecture. Use logged metrics for
outputs and values that change during training.

Learn more:

- [Experiments overview](https://docs.wandb.ai/models/track)
- [Configure experiments](https://docs.wandb.ai/models/track/config)
- [Python SDK reference](https://docs.wandb.ai/models/ref/python)

## Use framework integrations

W&B integrates with common ML frameworks so you can log metrics, system stats,
artifacts, and model checkpoints with minimal code changes. Some integrations that W&B Models supports include:

- PyTorch
- Keras
- TensorFlow
- HuggingFace Transformers
- PyTorch Lightning
- XGBoost

See [Integrations](https://docs.wandb.ai/models/integrations) in the W&B Developer guide for more information.

If a framework is not listed here, start with the
[Add W&B to a Python library](https://docs.wandb.ai/models/integrations/add-wandb-to-any-library) guide or use the core W&B Python SDK shown in
the previous section.

## Optimize hyperparameters with Sweeps

Use W&B Sweeps to define a hyperparameter search space, launch agents, and
compare results in the W&B App. Sweeps support search strategies such as grid
search, random search, and Bayesian optimization.

Good starting points:

- [Sweeps overview](https://docs.wandb.ai/models/sweeps)
- [Sweeps walkthrough](https://docs.wandb.ai/models/sweeps/walkthrough)
- [PyTorch Sweeps Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)
- [`examples/wandb-sweeps/`](examples/wandb-sweeps/)

## Version datasets and models with Artifacts

Use W&B Artifacts to track datasets, models, and other files as versioned inputs
and outputs of your runs. Artifacts help you reproduce experiments, inspect
lineage, and share assets across teams.

```python
import wandb

with wandb.init(project="artifact-demo", job_type="train") as run:
    model_artifact = wandb.Artifact("model", type="model")
    model_artifact.add_file("model.pt")
    run.log_artifact(model_artifact)
```

Learn more:

- [Artifacts overview](https://docs.wandb.ai/models/artifacts)
- [Create an artifact](https://docs.wandb.ai/models/artifacts/construct-an-artifact)
- [Download and use artifacts](https://docs.wandb.ai/models/artifacts/download-and-use-an-artifact)
- [Artifacts Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb)

## Visualize and query data with Tables

Use W&B Tables to log structured data, rich media, predictions, and evaluation
results. Tables are useful for comparing examples across models, finding
misclassifications, and sharing qualitative results.

```python
import wandb

with wandb.init(project="table-demo") as run:
    table = wandb.Table(
        columns=["id", "prediction", "label"],
        data=[
            [0, "cat", "cat"],
            [1, "dog", "cat"],
        ],
    )
    run.log({"predictions": table})
```

Learn more:

- [Tables overview](https://docs.wandb.ai/models/tables)
- [Log tables](https://docs.wandb.ai/models/tables/log_tables)
- [Tables quickstart Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb)
- [`colabs/tables/`](colabs/tables/)

## Share insights with Reports

Use W&B Reports to organize charts, describe findings, and share experiment
updates with collaborators. Reports can include plots, tables, media, Markdown,
and links to the underlying runs.

Learn more:

- [Reports overview](https://docs.wandb.ai/models/reports)
- [Create a report](https://docs.wandb.ai/models/reports/create-a-report)
- [Share reports](https://docs.wandb.ai/models/reports/collaborate-on-reports)