# Model Evaluation with Weights & Biases

This is a simple example that covers how to use W&B for:
1. Dataset Registry: Generate training dataset and register it with Weights & Biasespip 
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

Train a model based on the latest available dataset for the given model use case.

```shell
python model_trainer.py
```

Next, tweak some hyperparameters and re-run the model. You'll be able to compare
training performance for different models in the W&B dashboard. 

For example:
```shell
python model_trainer.py --validation_split 0.05
```


### 4. Evaluate candidate models

This script represents a workload that:
1. Finds all models that haven't yet been evaluated on the latest evaluation dataset
2. Runs the evaluation job for each model
3. Labels the best model "production" to feed into an inference system