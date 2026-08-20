# /// script
# dependencies = ["pandas", "scikit-learn", "torch", "ucimlrepo", "wandb"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{zoo-wandb} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{zoo-wandb} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Weights & Biases Registry Tutorial
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The goal of this notebook is to demonstrate how you and other members of your organization can use W&B Registry to track, share, and use dataset and model artifacts in your machine learning workflows. By the end of this notebook, you will know how to use W&B to:

    1. Create [collections](https://docs.wandb.ai/guides/registry/create_collection) within [W&B Registry](https://docs.wandb.ai/guides/registry)
    2. Make dataset and model artifacts available to other members of your organization, and
    3. Download your trained model and dataset artifacts from the registry for inference

    To achieve this, we will train a neural network to identify animal classes (mammal, amphibian, reptile, and so forth) based on features such as weather or not they ahve feathers, fins, and so on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Install and import packages
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb torch ucimlrepo scikit-learn pandas !pip install wandb torch ucimlrepo scikit-learn pandas
    return


@app.cell
def _():
    import torch 
    from torch import nn
    import pandas as pd
    import wandb
    from ucimlrepo import fetch_ucirepo

    from sklearn.model_selection import train_test_split

    return fetch_ucirepo, nn, pd, torch, train_test_split, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Retrieve and process dataset
    We will use the open source [Zoo dataset](https://archive.ics.uci.edu/dataset/111/zoo) from the UCI Machine Learning Repository.

    ### Retrieve dataset
    We can either manually download the dataset or use the [`ucimlrepo` package](https://github.com/uci-ml-repo/ucimlrepo) to import the dataset directly into our notebook. For this example, we will import the dataset directly into this notebook:
    """)
    return


@app.cell
def _(fetch_ucirepo):
    # fetch dataset 
    zoo = fetch_ucirepo(id=111) 
  
    # data (as pandas dataframes) 
    X = zoo.data.features 
    y = zoo.data.targets
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Explore the data

    Let's take a quick look at the shape and data type of the dataset:
    """)
    return


@app.cell
def _(X, y):
    print("features: ", X.shape, "type: ", type(X))
    print("labels: ", y.shape, "type: ", type(y))
    return


@app.cell
def _(X):
    X.head(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Process data

    For training let's convert our dataset from a [pandas `DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) to [a tensor with PyTorch](https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor), convert the data type of our input tensor(float64 to float32) to match the data type of the `nn.Linear module`, and convert our label tensor to index from 0-6:
    """)
    return


@app.cell
def _(X, torch, y):
    # Data type of the data must match the data type of the model, the default dtype for nn.Linear is torch.float32
    dataset = torch.tensor(X.values).type(torch.float32) 

    # Convert to tensor and format labels from 0 - 6 for indexing
    labels = torch.tensor(y.values)  - 1

    print("dataset: ", dataset.shape, "dtype: ",dataset.dtype)
    print("labels: ", labels.shape, "dtype: ",labels.dtype)
    return dataset, labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Save processed dataset locally using [`torch.save`](https://pytorch.org/docs/stable/generated/torch.save.html)
    """)
    return


@app.cell
def _(dataset, labels, torch):
    torch.save(dataset, "zoo_dataset.pt")
    torch.save(labels, "zoo_labels.pt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Track and publish dataset

    Within the Dataset registry we will create a collection called "zoo-dataset-tensors". A *collection* is a set of linked artifact versions in a registry.

    To create a collection we need to do two things:
    1. Specify the collection and registry we want to link our artifact version to. To do this, we specify a "target path" for our artifact version.
    2. Use the `wandb.run.link_artifact` method and pass our artifact object and the target path.

    #### Define target path of the collection

    The target path of a collection consists of two parts:
    * The name of the registry
    * The name of the collection within the registry

    If you know these two fields, you can create the full name yourself with string concatenation, f-strings, and so forth:
    ```python
    target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Publish dataset to registry

    Let's publish our dataset to the Dataset registry in a collection called "zoo-dataset-tensors". To do this, we will

    1. Get or create the target path. (For this notebook we will programmatically create the target path).
    1. Initialize a run
    1. Create an artifact object
    2. Add each split dataset as individual files to the artifact object
    3. Link the artifact object to the collection with `run.link_artifact()`. Here we specify the target path and the artifact we want to link.

    First, let's create the target path:
    """)
    return


@app.cell
def _():
    REGISTRY_NAME = "Dataset"
    COLLECTION_NAME = "zoo-dataset-tensors"

    # Path to link the artifact to a collection
    dataset_target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
    return (dataset_target_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have the target path, let's publish the dataset to the "Dataset" registry. In the following code cell, ensure to replace the values enclosed in `<>` with your team's entity:
    """)
    return


@app.cell
def _(dataset_target_path, wandb):
    TEAM_ENTITY = "<TEAM_A>"
    PROJECT = "zoo_experiment"

    run = wandb.init(
        entity=TEAM_ENTITY,
        project=PROJECT,
        job_type="publish_dataset"
    )

    artifact = wandb.Artifact(
        name="zoo_dataset",
        type="dataset",  
        description="Processed dataset and labels."
    )

    artifact.add_file(local_path="zoo_dataset.pt", name="zoo_dataset")
    artifact.add_file(local_path="zoo_labels.pt", name="zoo_labels")

    run.link_artifact(artifact=artifact, target_path=dataset_target_path)

    run.finish()
    return PROJECT, TEAM_ENTITY


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Split data and publish split dataset to registry
    Split the data into a training and test set. Splitting the dataset and tracking them as separate files will make it easier for a different user to use the same datasets for future reproducibility, testing, and analysis.
    """)
    return


@app.cell
def _():
    # Describe how we split the training dataset for future reference, reproducibility.
    config = {
        "random_state" : 42,
        "test_size" : 0.25,
        "shuffle" : True
    }
    return (config,)


@app.cell
def _(config, dataset, labels, torch, train_test_split):
    # Split dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        dataset,labels, 
        random_state=config["random_state"],
        test_size=config["test_size"], 
        shuffle=config["shuffle"]
    )

    # Save the files locally
    torch.save(X_train, "zoo_dataset_X_train.pt")
    torch.save(y_train, "zoo_labels_y_train.pt")

    torch.save(X_test, "zoo_dataset_X_test.pt")
    torch.save(y_test, "zoo_labels_y_test.pt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, let's publish this dataset into a different collection within the Dataset registry called "zoo-dataset-tensors-split":
    """)
    return


@app.cell
def _(PROJECT, TEAM_ENTITY, config, wandb):
    run_1 = wandb.init(entity=TEAM_ENTITY, project=PROJECT, job_type='publish_split_dataset', config=config)
    artifact_1 = wandb.Artifact(name='split_zoo_dataset', type='dataset', description='Artifact contains `zoo_dataset` split into 4 datasets.                 For training, use `zoo_dataset_X_train` and `zoo_labels_y_train`.                 For testing, use `zoo_dataset_X_test` and `zoo_labels_y_test`.')
    artifact_1.add_file(local_path='zoo_dataset_X_train.pt', name='zoo_dataset_X_train')
    artifact_1.add_file(local_path='zoo_labels_y_train.pt', name='zoo_labels_y_train')
    artifact_1.add_file(local_path='zoo_dataset_X_test.pt', name='zoo_dataset_X_test')
    artifact_1.add_file(local_path='zoo_labels_y_test.pt', name='zoo_labels_y_test')
    REGISTRY_NAME_1 = 'Dataset'
    # Let's add a description to let others know which file to use in future experiments
    COLLECTION_NAME_1 = 'zoo-dataset-tensors-split'
    target_dataset_path = f'wandb-registry-{REGISTRY_NAME_1}/{COLLECTION_NAME_1}'
    run_1.link_artifact(artifact=artifact_1, target_path=target_dataset_path)
    # Create a target path for our artifact in the registry
    run_1.finish()
    return COLLECTION_NAME_1, REGISTRY_NAME_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can verify we correctly linked our artifact to our desired collection and registry with W&B App UI:

    1. Navigate to the Registry App
    2. Select on the Dataset registry
    3. Click **View details** "zoo-dataset-tensors-split" collection
    4. Click the **View** button next to the artifact version
    5. Select the **Files** tab

    You should see four files: "zoo_dataset_X_test", "zoo_dataset_X_train", "zoo_labels_y_test", and "zoo_labels_y_train".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define a model

    The following cells show how to create a simple neural network classifier with PyTorch. There is nothing unique about this model, so we'll will not go into detail of this code block.
    """)
    return


@app.cell
def _(nn):
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_stack = nn.Sequential(
                nn.Linear(in_features=16 , out_features=16),
                nn.Sigmoid(),
                nn.Linear(in_features=16, out_features=7)
            )

        def forward(self, x):
            logits = self.linear_stack(x)
            return logits

    model = NeuralNetwork()
    print(model)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define hyperparameters, loss function, and optimizer
    """)
    return


@app.cell
def _():
    hyperparameter_config = {
        "learning_rate": 0.1,
        "epochs": 1000,
        "model_type": "Multivariate_neural_network_classifier",
    }
    return (hyperparameter_config,)


@app.cell
def _(hyperparameter_config, model, nn, torch):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=hyperparameter_config["learning_rate"])
    return loss_fn, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train model

    Next, let's train a model using the training data we published to the registry earlier in this notebook. After we train the model, we will publish that model to W&B.

    To do this, let's first get the artifact we published to the "Dataset" registry. To retrieve an artifact from a registry, we need to know the name of that artifact. The name of an artifact in a registry consists of the prefix `wandb-registry-`, the name of the registry, the name of the collection, and the artifact version:

    ```python
    # Artifact name/filepath for downloading and using artifacts published to a registry
    f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"
    ```
    Since we only linked one artifact version, the version we'll use is `v0`. (W&B uses 0 indexing).

    Note that the name of an artifact is nearly identical to the target path we specified in a previous step when we publish our artifact to the registry except for the version number:

    ```python
    # Target path for publising an artifact version to a registry
    f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
    ```
    """)
    return


@app.cell
def _(
    COLLECTION_NAME_1,
    PROJECT,
    REGISTRY_NAME_1,
    TEAM_ENTITY,
    hyperparameter_config,
    loss_fn,
    model,
    optimizer,
    torch,
    wandb,
):
    run_2 = wandb.init(entity=TEAM_ENTITY, project=PROJECT, job_type='training', config=hyperparameter_config)
    VERSION = 0
    # Get dataset artifacts from registry
    artifact_name = f'wandb-registry-{REGISTRY_NAME_1.lower()}/{COLLECTION_NAME_1}:v{VERSION}'
    dataset_artifact = run_2.use_artifact(artifact_or_name=artifact_name)
    X_train_path = dataset_artifact.download(path_prefix='zoo_dataset_X_train')
    y_train_path = dataset_artifact.download(path_prefix='zoo_labels_y_train')
    # Download only the training data
    X_train_1 = torch.load(f=X_train_path + '/zoo_dataset_X_train')
    y_train_1 = torch.load(f=y_train_path + '/zoo_labels_y_train')
    prev_best_loss = 10000000000.0
    # Load data as tensors 
    for e in range(hyperparameter_config['epochs'] + 1):
        pred = model(X_train_1)
        loss = loss_fn(pred, y_train_1.squeeze(1))
    # Set initial dummy loss value to compare to in training loop
        loss.backward()
        optimizer.step()
    # Training loop
        optimizer.zero_grad()
        wandb.log({'train/epoch_ndx': e, 'train/train_loss': loss})
        if e % 100 == 0 and loss <= prev_best_loss:
            print('epoch: ', e, 'loss:', loss.item())
            PATH = 'zoo_wandb.pth'
            torch.save(model.state_dict(), PATH)
            model_artifact_name = f'zoo-{wandb.run.id}'
            artifact_2 = wandb.Artifact(name=model_artifact_name, type='model', metadata={'num_classes': 7, 'model_type': wandb.config['model_type']})
            prev_best_loss = loss
    print(f'Saving model artifact {model_artifact_name}')
    artifact_2.add_file(PATH)
    artifact_2.save()
    # Add saved model to artifact
    run_2.finish()  # Checkpoint/save model if loss improves  # Store new best loss
    return (model_artifact_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The preceding cell might look intimidating. Let's break it down:

    * First, we download the dataset from the Dataset registry and load it as a tensor
    * Next, we create a simple training loop
      * Within the training loop we log the loss for each step
      * We checkpoint(save) the model every time the remainder of the epoch divided by 100 is 0 and the loss is lower than the previously recorded loss.
      * We then add the saved PyTorch model to our artifact object.

    A couple of things to note:
    1. The preceding code cell adds a single artifact version to W&B. You can confirm this by navigating to your project workspace, select **Artifacts** in the left navigation, and under **models** click the name of the artifact (starts with `zoo-{run.id}`). You will see a single model with version `v0`.
    2. At this point, we have only tracked the model artifact within our team's project. Anyone outside of our team does not have access to the model we created. To make this model accessible to members outside of our team, we will need to publish our model to the registry.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Publish model to registry
    Let's make this model artifact available to other users in our organization. To do this, we will create a collection within the Model registry.

    To create a collection within a registry, we need to know the full name of the artifact. The full name of the artifact consists of the name we provided to it when we created the Artifact object and its location within our team's project.

    There are two ways to get the full name of an artifact, interatively with the W&B App UI or programmatically with the W&B Python SDK. In this example, we'll programmatically create the name of the artifact since we have these values loaded into memory.

    ### Programmatically create name of artifact

    The full name of an artifact consists of four components:
    * Team entity
    * Project name
    * The name of the artifact (the string you passed when you created the artifact object with `wandb.Artifact()`)
    * The artifact version

    Putting this together, the full name of an artifact is:
    ```python
    # Full name of an artifact in a team project
    artifact_name = f'{TEAM_ENTITY}/{PROJECT}/{ARTIFACT_NAME}:v{VERSION}'
    ```
    """)
    return


@app.cell
def _(PROJECT, TEAM_ENTITY, model_artifact_name):
    # Artifact name specifies the specific artifact version within our team's project
    artifact_name_1 = f'{TEAM_ENTITY}/{PROJECT}/{model_artifact_name}:v0'
    print('Artifact name: ', artifact_name_1)
    return (artifact_name_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have the full name of our model artifact. Let's publish it to the model registry.

    Similar to how we created a target path when we published our dataset artifact to the Dataset registry, let's create the target path for our model artifact. The target path tells W&B the collection and registry (Model registry) to link our artifact version to.

    As a reminder, the target path to link an artifact to a registry consists of:

    ```python
    # Target path used to link artifact to registry
    target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
    ```
    """)
    return


@app.cell
def _():
    REGISTRY_NAME_2 = 'Model'
    COLLECTION_NAME_2 = 'Zoo_Classifier_Models'
    target_path = f'wandb-registry-{REGISTRY_NAME_2}/{COLLECTION_NAME_2}'
    print('Target path: ', target_path)
    return (target_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Putting this all together, we specify our artifact name in `run.use_artifact()` and the target path for `run.link_artifact()`:
    """)
    return


@app.cell
def _(PROJECT, TEAM_ENTITY, artifact_name_1, target_path, wandb):
    run_3 = wandb.init(entity=TEAM_ENTITY, project=PROJECT)
    model_artifact = run_3.use_artifact(artifact_or_name=artifact_name_1, type='model')
    run_3.link_artifact(artifact=model_artifact, target_path=target_path)
    run_3.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The preceding code block links our model artifact version to a collection called "Zoo_Classifier_Models" within the model registry.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### View lineage map of registered model

    Let's say that you did not know exactly which model version to use. You can check the lineage of all artifact versions on the W&B App UI. The lineage shows which artifacts were used as input to a run and which artifacts were the output of a given run.

    For example, the image below shows the "Zoo_Classifier_Models" collection within the model registry. Highlighted in yellow is the current model artifact version that is linked to the registry.

    From left to right we see that the run "trim-rain-2" was responible for creating the "split_zoo_dataset" artifact. (Recall that this is the dataset artifact that contains the test and training data).

    We then see that the "golden-sunset-3" run used the "split_zoo_dataset" artifact for training. Within this run, we created a model artifact. The specific artifact version we linked to "Zoo_Classifier_Models" is called `zoo-wyhak4o0:v10`.

    ![](./images/dag_model_registry.png)

    To view the lineage map of an artifact in a registry:

    1. Navigate to the Registry app at https://wandb.ai/registry
    2. Click on a registry
    3. Select an artifact version
    3. Select the **Lineage** tab
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Download artifacts from registry for inference

    For this last section, suppose you are a different user in a different team within the same organization. You and your team want to download the model and test dataset that was published to your organization's registry by a different team. You and your team will use the model and test dataset for inference and store those findings in a project called "Check_Zoo_Model".

    Note: The team member that wants do use and download published artifacts has [member role permissions](https://docs.wandb.ai/guides/registry/configure_registry#registry-roles-permissions).  This means they can view and download artifacts from the registry.

    How can you retrieve the artifacts version that were published by another team? Simple:

    1. Get the full name of the artifact version programmatically or interactively with the W&B App UI
    2. Use the W&B Python SDK to download the artifacts

    #### Interactively get full name of model and dataset artifacts from registry
    1. Go to the W&B Registry app at https://wandb.ai/registry/.
    2. Select the registry that your artifact is linked to.
    3. Click the **View details** button next to the name of the collection with your linked artifact.
    4. Click on the **View** button next to the artifact version.
    5. Within the **Version** tab, copy path listed next to **Full Name**.
    6. Paste the full name of the registry for the `artifact_or_name` field in `run.use_artifact()`.

    Note: In this example, we happen to know these values, so we'll programmatically create the full name of our model and dataset artifacts published in the registry.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Get trained model from model registry
    """)
    return


@app.cell
def _():
    # Create model artifact name
    REGISTRY_NAME_3 = 'model'
    COLLECTION_NAME_3 = 'Zoo_Classifier_Models'
    VERSION_1 = 0
    model_artifact_name_1 = f'wandb-registry-{REGISTRY_NAME_3}/{COLLECTION_NAME_3}:v{VERSION_1}'
    print(f'Model artifact name: {model_artifact_name_1}')
    return (model_artifact_name_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the following code cell, ensure to replace the values enclosed in `<>` with the entity of a different team in your organization than the one you specified earlier in this notebook.

    Note: If you do not have another team entity, you can re-use the entity you specified earlier.
    """)
    return


@app.cell
def _(model_artifact_name_1, wandb):
    # Enter information about your team and your team's project
    DIFFERENT_TEAM_ENTITY = '<TEAM_B>'
    DIFFERENT_PROJECT = 'Check_Zoo_Model'
    run_4 = wandb.init(entity=DIFFERENT_TEAM_ENTITY, project=DIFFERENT_PROJECT)
    registry_model = run_4.use_artifact(artifact_or_name=model_artifact_name_1)
    local_model_path = registry_model.download()
    return DIFFERENT_PROJECT, DIFFERENT_TEAM_ENTITY, local_model_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For PyTorch models, we need to redefine our model architecture:
    """)
    return


@app.cell
def _(local_model_path, nn, torch):
    class NeuralNetwork_1(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear_stack = nn.Sequential(nn.Linear(in_features=16, out_features=16), nn.Sigmoid(), nn.Linear(in_features=16, out_features=7))

        def forward(self, x):
            logits = self.linear_stack(x)
            return logits
    loaded_model = NeuralNetwork_1()
    loaded_model.load_state_dict(torch.load(f=local_model_path + '/zoo_wandb.pth'))
    return (loaded_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Get test dataset from Dataset registry

    Let's get the test dataset from our registry. Similar to the above code block, we will specify the full name of the artifact version we want from our Dataset registry.
    """)
    return


@app.cell
def _():
    # Create dataset artifact name
    REGISTRY_NAME_4 = 'dataset'
    COLLECTION_NAME_4 = 'zoo-dataset-tensors-split'
    VERSION_2 = 0
    data_artifact_name = f'wandb-registry-{REGISTRY_NAME_4}/{COLLECTION_NAME_4}:v{VERSION_2}'
    print(f'Dataset artifact name: {data_artifact_name}')
    return (data_artifact_name,)


@app.cell
def _(DIFFERENT_PROJECT, DIFFERENT_TEAM_ENTITY, data_artifact_name, wandb):
    run_5 = wandb.init(entity=DIFFERENT_TEAM_ENTITY, project=DIFFERENT_PROJECT)
    dataset_artifact_1 = run_5.use_artifact(artifact_or_name=data_artifact_name, type='dataset')
    local_dataset_path = dataset_artifact_1.download()
    return local_dataset_path, run_5


@app.cell
def _(local_dataset_path, run_5, torch):
    # Test data and label filenames
    test_data_filename = 'zoo_dataset_X_test'
    test_labels_filename = 'zoo_labels_y_test'
    loaded_data = torch.load(f'{local_dataset_path}/{test_data_filename}')
    # Load dataset and labels into notebook
    loaded_labels = torch.load(f'{local_dataset_path}/{test_labels_filename}')
    run_5.finish()
    return loaded_data, loaded_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Make predictions with loaded model

    How does our model perform? Recall that the goal of the neural network is to predict the animall class based on features of that animal.

    For each prediction, our model returns an integer that refers the class. Let's create a dictionary so we can map the integer to the class name:
    """)
    return


@app.cell
def _():
    class_labels = {
        0: "Aves",
        1: "Mammalia",
        2: "Reptilia",
        3: "Actinopterygii",
        4: "Amphibia",
        5: "Insecta",
        6: "Crustacea",
    }
    return (class_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's feed our model some data to make predictions:
    """)
    return


@app.cell
def _(loaded_data, loaded_model, torch):
    outputs = loaded_model(loaded_data)
    __, predicted = torch.max(outputs, 1)
    return (predicted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These integers don't mean much, let's convert them to return the animal class and store this into a pandas DataFrame for us to compare the predicted vs the true values:
    """)
    return


@app.cell
def _(class_labels, loaded_labels, pd, predicted):
    results = list(map(lambda x: class_labels.get(x), predicted.numpy()))
    true_values = list(map(lambda x: class_labels.get(x), loaded_labels.squeeze().numpy()))

    # Create pandas DataFrame
    df = pd.DataFrame(
        {
            'Predicted': results,
            'True values': true_values
        }
    )

    # Create new column where we compare the predicted vs true
    df["Predicted correctly"] = df["Predicted"] == df["True values"]
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how many it predicted corrected
    """)
    return


@app.cell
def _(df):
    # Count how many predictions were wrong
    df['Predicted correctly'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's view these as percentages:
    """)
    return


@app.cell
def _(df):
    # Get percentage 
    df['Predicted correctly'].value_counts(normalize=True).mul(100).astype(str)+'%'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The percentage the model predicted correct might vary. As of writing this notebook, our the model correctly predicted ~88% of the time:
    """)
    return


app._unparsable_cell(
    r"""
    Predicted correctly
    True      88.46153846153845%
    False    11.538461538461538%
    Name: proportion, dtype: object
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As a next step, you could dig into the examples that were incorrectly predicted and try to figure out why it predicted incorrectly. You could also try feature engineering to extract more features to train with.

    ## Summary
    In this notebook completed each major step in a typical machine learning workflow, from downloading a dataset, processing the dataset, defining a model, training that model on processed data, checking/saving the best model, and checking how the model performed by making predictions with that model on a dataset it had not seen before.

    Throughout the process you learned how to use W&B Registry to:

    * Track and publish multiple datasets
    * Track and publish a model
    * Mimic how someone else in your organization (with correct permission) can download and use model and datasets published in the W&B Registry for further analysis.

    ## Next steps:

    As the number of machine learning experiments increases, so does the complexity of keeping track of saved models and datasets. For each model version, we recommend that you document key aspects of your model such as a brief summary of the model, information about the architecture of the model, how someone can deserialize a saved model, and so forth. You can provide all of this information, and more, within the **Description** field of the model version.
    """)
    return


if __name__ == "__main__":
    app.run()
