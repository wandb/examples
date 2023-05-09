# Model Evaluation with Weights & Biases

This is a simple example that covers how to use W&B for:
1. Dataset Registry: Generate training dataset and register it with Weights & Biases
2. Model Registry: Train models on the dataset, and register resulting model files
3. Model Evaluation: Evaluate models from the model registry, and tag production-ready models

### 1. Install requirements

Install the Weights & Biases library `wandb` and other dependencies.
```shell
pip install -r requirements.txt
```


### 2. Register a dataset

Generate and register a dataset for a particular model use case. In this example,
we use the MNIST dataset for simplicity.

```shell
python dataset_generator.py
```


### 3. Train some models

Train a model based on the latest available dataset for the given model use case. Change
hyperparameters to get different versions of the model.

```shell
python model_trainer.py
python model_trainer.py --validation_split 0.05
python model_trainer.py --batch_size 64
```

### 4. Evaluate candidate models

Next, run a model evaluation job that:
1. Finds all models that haven't yet been evaluated on the latest evaluation dataset
2. Runs the evaluation job for each model
3. Labels the best model "production" to feed into an inference system

```shell
python model_evaluator.py
```

### 5. Visualize results

Create tables to visualize your results. Here's [an example report](https://wandb.ai/timssweeney/model_registry_example/reports/MNIST-Model-Status--Vmlldzo4OTIyNTA) that captures and compares trained models.

Learn more in the [Model Management docs](https://docs.wandb.ai/guides/models).
