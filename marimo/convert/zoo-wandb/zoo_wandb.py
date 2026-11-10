# /// script
# dependencies = [
#     "marimo>=0.25.2",
#     "pandas",
#     "scikit-learn",
#     "torch",
#     "ucimlrepo",
#     "wandb",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/zoo-wandb/zoo_wandb.py/server)

    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
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
    Use Weights & Biases Registry to share versioned datasets and models across
    teams. This notebook converts the UCI Zoo dataset to tensors, logs and links
    those files to a Registry collection, trains a classifier, and retrieves the
    saved model and test data for inference.

    The classifier uses animal attributes such as feathers, fins, and number of
    legs to predict one of seven dataset classes. The model is intentionally
    small so the notebook can focus on Artifact paths, version selection, and lineage.
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
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def _notify_submission(_value):
        mo.status.toast(
            "Credentials submitted",
            "Run the next cell to verify them with Weights & Biases.",
        )

    wandb_api_form = (
        mo.md("{api_key}")
        .batch(
            api_key=mo.ui.text(
                kind="password",
                label="Forge API key",
                full_width=True,
            )
        )
        .form(
            submit_button_label="Connect to Weights & Biases",
            bordered=True,
            on_change=_notify_submission,
        )
    )

    wandb_api_form
    return (wandb_api_form,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After submitting the API key, run the following cell to verify the
    credentials and load your available Weights & Biases entities.
    """)
    return


@app.cell(hide_code=True)
def _(mo, wandb, wandb_api_form):
    from wandb.sdk.lib.service.service_connection import (
        WandbApiFailedError as _WandbApiFailedError,
    )

    mo.stop(wandb_api_form.value is None)

    _submission = wandb_api_form.value
    _api_key = _submission["api_key"].strip()

    try:
        _login_ok = wandb.login(
            key=_api_key or None,
            relogin=bool(_api_key),
            verify=True,
            prompt=False,
        )
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout("Check your API key or saved credentials, then connect again.", kind="danger"),
    )

    try:
        _api = wandb.Api(timeout=15)
        _viewer = _api.viewer
    except (wandb.errors.Error, _WandbApiFailedError, ValueError):
        mo.stop(
            True,
            mo.callout(
                "Weights & Biases verified the credentials but could not retrieve your account entities.",
                kind="danger",
            ),
        )

    available_entities = sorted({
        _value
        for _value in (
            getattr(_viewer, "username", None),
            getattr(_viewer, "entity", None),
            *_viewer.teams,
        )
        if _value
    })
    mo.stop(
        not available_entities,
        mo.callout(
            "Weights & Biases authenticated successfully, but no personal or team entities were found.",
            kind="danger",
        ),
    )

    mo.callout(
        mo.md("✓ Credentials verified. Specify your team entity below."),
        kind="success",
    )
    return (available_entities,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set your team and project

    Run the following cell and specify your:

    - Weights & Biases team entity.
    - A name for your [project](https://docs.coreweave.com/products/wandb/track/project-page#projects). The default project is `zoo_experiment`.

    Then select **Use this Weights & Biases target**.

    > If the project doesn't exist, Weights & Biases creates it when the first run starts, provided you have write access.
    """)
    return


@app.cell(hide_code=True)
def _(available_entities, mo):
    def _validate_wandb_target(value):
        if value is None:
            return None
        if not value["entity"]:
            return "Select a Weights & Biases entity."
        if not value["project"].strip():
            return "Enter a project name."
        return None

    def _notify_wandb_submission(_value):
        mo.status.toast(
            "Weights & Biases target selected",
            "Run the next cell to confirm the destination.",
        )

    wandb_setup_form = (
        mo.md("{entity}\n\n{project}")
        .batch(
            entity=mo.ui.dropdown(
                options=available_entities,
                value=available_entities[0],
                label="Weights & Biases entity",
                searchable=True,
                full_width=True,
            ),
            project=mo.ui.text(
                value="zoo_experiment",
                label="Project",
                full_width=True,
            ),
        )
        .form(
            submit_button_label="Use this Weights & Biases target",
            validate=_validate_wandb_target,
            on_change=_notify_wandb_submission,
            bordered=True,
        )
    )

    wandb_setup_form
    return (wandb_setup_form,)




@app.cell(hide_code=True)
def _(mo, wandb_setup_form):
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _entity = _submission["entity"]
    _project = _submission["project"].strip()

    TEAM_ENTITY = _entity
    PROJECT = _project

    return PROJECT, TEAM_ENTITY


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create a registry

    Next, [create a registry](https://docs.coreweave.com/products/registry/create_registry#create-a-registry) with the Weights & Biases Public API:
    """)
    return


@app.cell
def _():
    REGISTRY_NAME = "New_Zoo_Registry"

    # wandb.Api().create_registry(
    #     name=REGISTRY_NAME,
    #     visibility="organization",
    # )
    return (REGISTRY_NAME,)


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
    ### Convert the data to tensors

    The following cell converts feature values to `float32`, matching the linear layers’ weights.

    We also convert labels to integer tensors and subtract one so that the original class
    `IDs 1–7` become `0–6`, as required by the seven-output classifier.
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
    ## Log and link the dataset

    An [Artifact](https://docs.coreweave.com/products/wandb/artifacts) stores a versioned set of files in a project. A [registry collection](https://docs.coreweave.com/products/registry)
    contains links to Artifact versions so authorized users can discover and reuse
    them. [Linking an artifact](https://docs.coreweave.com/products/registry/link_version) creates a reference to the source version.

    A _target path_ identifies the registry and collection.

    This example uses a `dataset` registry and a collection named
    `zoo-dataset-tensors`. Match the registry name to the one in your organization.

    ```python
    target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
    ```

    See [Link an artifact version to a collection](https://docs.coreweave.com/products/registry/link_version)
    for the required permissions and supported paths.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Publish the dataset to Registry

    The following cell define the target path of the dataset:
    """)
    return


@app.cell
def _(REGISTRY_NAME):
    COLLECTION_NAME = "zoo-dataset-tensors"

    # Path to link the artifact to a collection
    dataset_target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
    return (dataset_target_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next cell initializes a run, creates an artifact object, adds `zoo_dataset.pt` and `zoo_labels.pt` to a dataset Artifact and then links the artifact version to registry (defined by `dataset_target_path`).

    > These files contain the full dataset. The following section creates
    and publishes separate training and test files.
    """)
    return


@app.cell
def _(PROJECT, TEAM_ENTITY, dataset_target_path, wandb):
    with wandb.init(
        entity=TEAM_ENTITY,
        project=PROJECT,
        job_type="publish_dataset"
    ) as run:

        artifact = wandb.Artifact(
            name="zoo_dataset",
            type="dataset",  
            description="Processed dataset and labels."
        )
    
        artifact.add_file(local_path="zoo_dataset.pt", name="zoo_dataset")
        artifact.add_file(local_path="zoo_labels.pt", name="zoo_labels")
    
        run.link_artifact(artifact=artifact, target_path=dataset_target_path)

    return


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
    Next, let's publish this dataset into a different collection called "zoo-dataset-tensors-split":
    """)
    return


@app.cell
def _(PROJECT, REGISTRY_NAME, TEAM_ENTITY, config, wandb):
    job_type = 'publish_split_dataset'
    name = 'split_zoo_dataset'

    with wandb.init(entity=TEAM_ENTITY, project=PROJECT, job_type=job_type, config=config) as run_1:

        description = 'Contains `zoo_dataset` split into 4 datasets. ' \
        'For training, use `zoo_dataset_X_train` and `zoo_labels_y_train ' \
        'For testing, use `zoo_dataset_X_test` and `zoo_labels_y_test`.'
    
        artifact_1 = wandb.Artifact(name=name, type='dataset', description=description)
        artifact_1.add_file(local_path='zoo_dataset_X_train.pt',name='zoo_dataset_X_train')
        artifact_1.add_file(local_path='zoo_labels_y_train.pt', name='zoo_labels_y_train')
        artifact_1.add_file(local_path='zoo_dataset_X_test.pt', name='zoo_dataset_X_test')
        artifact_1.add_file(local_path='zoo_labels_y_test.pt', name='zoo_labels_y_test')
    
    
        # Add a description to let others know which file to use in future experiments
        COLLECTION_NAME_1 = 'zoo-dataset-tensors-split'
        target_dataset_path = f'wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME_1}'
        run_1.link_artifact(artifact=artifact_1, target_path=target_dataset_path)

    return (COLLECTION_NAME_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can verify we correctly linked our artifact to our desired collection and registry with Weights & Biases App UI:

    1. Navigate to the Registry App
    2. Select the registry
    3. Click **View details** "zoo-dataset-tensors-split" collection
    4. Click the **View** button next to the artifact version
    5. Select the **Files** tab

    You should see four files: **zoo_dataset_X_test**, **zoo_dataset_X_train**, **zoo_labels_y_test**, and **zoo_labels_y_train**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define a classifier

    The next cell defines a classifer using PyTorch. The network maps 16 animal attributes to 16 hidden units with a sigmoid activation, then produces seven class scores. Cross-entropy loss accepts these scores directly.

    > The same architecture must be recreated when loading the saved state dictionary later.
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
    ## Train the model

    Retrieve the split dataset through its registry path so the training run
    records exactly which Artifact version it consumed.

    ```python
    artifact_name = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"
    ```

    The code below selects `v0`. If the collection already existed or you reran publication, inspect its versions and update `VERSION` before training.

    Recall that the target path identifies the collection used when linking. Adding a version suffix selects a specific entry to download.
    """)
    return


@app.cell
def _(
    COLLECTION_NAME_1,
    PROJECT,
    REGISTRY_NAME,
    TEAM_ENTITY,
    hyperparameter_config,
    loss_fn,
    model,
    optimizer,
    torch,
    wandb,
):
    with wandb.init(entity=TEAM_ENTITY, project=PROJECT, job_type='training', config=hyperparameter_config) as run_2:

        VERSION = 0
   
        # Get dataset artifacts from registry
        artifact_name = f'wandb-registry-{REGISTRY_NAME.lower()}/{COLLECTION_NAME_1}:v{VERSION}'
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
            run_2.log({'train/epoch_ndx': e, 'train/train_loss': loss})
            if e % 100 == 0 and loss <= prev_best_loss:
                print('epoch: ', e, 'loss:', loss.item())
                PATH = 'zoo_wandb.pth'
                torch.save(model.state_dict(), PATH)
                model_artifact_name = f'zoo-{run_2.id}'
                artifact_2 = wandb.Artifact(name=model_artifact_name, type='model', metadata={'num_classes': 7, 'model_type': run_2.config['model_type']})
                prev_best_loss = loss
    
        print(f'Saving model artifact {model_artifact_name}')
        artifact_2.add_file(PATH)
        artifact_2.save()
        # Add saved model to artifact
    return (model_artifact_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The training run downloads the feature and label files from the registered
    split, updates the model, and logs training loss. Every 100 iterations,
    including iteration zero, it saves weights when the loss improves over the
    previous saved checkpoint.

    The checkpoint file is overwritten locally. After training, the code uploads
    only the final saved checkpoint as one model Artifact version. It does not
    upload a version at every checkpoint interval, and checkpoint selection uses
    training loss rather than held-out performance.

    The model is now stored in the selected project. The next step links that
    version to the registry.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Link the trained model to the registry

    Let's link the model we trained in the previous to the registry.

    First defined the model artifact's name. It is composed of the team that created and logged the artifact, the project, artifact name, and the version:

    ```python
    artifact_name = f"{TEAM_ENTITY}/{PROJECT}/{ARTIFACT_NAME}:v{VERSION}"
    ```

    The following cell defines the artifact's name using the values defined earlier in this notebook:
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
    Next, define the destination collection and registry for the model version:

    ```python
    target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
    ```

    The source Artifact path selects the files to share. The target path selects where users will discover that version in Registry.
    """)
    return


@app.cell
def _(REGISTRY_NAME):
    COLLECTION_NAME_2 = 'Zoo_Classifier_Models'

    target_path_2 = f'wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME_2}'
    print('Target path: ', target_path_2)
    return (target_path_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Putting this all, let's now link the model to the registry.

    We pass the model artifact's name to `wandb.Run.use_artifact()` and we pass the target path to `wandb.Run.link_artifact()`:
    """)
    return


@app.cell
def _(PROJECT, TEAM_ENTITY, artifact_name_1, target_path_2, wandb):
    with wandb.init(entity=TEAM_ENTITY, project=PROJECT) as run_3:
        model_artifact = run_3.use_artifact(artifact_or_name=artifact_name_1, type='model')
        run_3.link_artifact(artifact=model_artifact, target_path=target_path_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inspect the registered model’s lineage

    Open the model version in Registry and inspect its **Lineage** view. Follow
    the input edge to the split dataset and the output edge from the training
    run to the model. This connects the saved weights to the data version used
    for training.

    Use the lineage graph to choose a model version based on its recorded inputs
    and training run, then copy that version’s full path for inference.

    ## Download registered artifacts for inference

    The final section simulates using the registered model from another team’s
    project. It downloads model weights and the test split, recreates the model,
    and compares predictions with the saved labels.

    Open each version in Registry and copy its full asrtifact path, or construct
    the path from the registry, collection, and version.

    The examples below use `VERSION_1 = 0` and `VERSION_2 = 0`. Update them if the intended model or test split has another version number.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Get trained model from model registry
    """)
    return


@app.cell
def _(REGISTRY_NAME):
    # Create model artifact name
    COLLECTION_NAME_3 = 'Zoo_Classifier_Models'
    VERSION_1 = 0

    model_artifact_name_1 = f'wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME_3}:v{VERSION_1}'

    print(f'Model artifact name: {model_artifact_name_1}')
    return (model_artifact_name_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following cell downloads the model we previously linked to the registry:
    """)
    return


@app.cell
def _(PROJECT, TEAM_ENTITY, model_artifact_name_1, wandb):
    with wandb.init(entity=TEAM_ENTITY, project=PROJECT) as run_4:
        registry_model = run_4.use_artifact(artifact_or_name=model_artifact_name_1)
        local_model_path = registry_model.download()
    return (local_model_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The checkpoint contains a state dictionary. Recreate the same architecture
    before loading its saved weights.
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
    ### Get test dataset from registry

    Let's get the test dataset from our registry. Similar to the above code block, we will specify the full name of the artifact version we want from our registry.
    """)
    return


@app.cell
def _(REGISTRY_NAME):
    # Create dataset artifact name
    COLLECTION_NAME_4 = 'zoo-dataset-tensors-split'
    VERSION_2 = 0

    data_artifact_name = f'wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME_4}:v{VERSION_2}'

    print(f'Dataset artifact name: {data_artifact_name}')
    return (data_artifact_name,)


@app.cell
def _(PROJECT, TEAM_ENTITY, data_artifact_name, torch, wandb):
    with wandb.init(entity=TEAM_ENTITY, project=PROJECT) as run_5:
        dataset_artifact_1 = run_5.use_artifact(artifact_or_name=data_artifact_name, type='dataset')
        local_dataset_path = dataset_artifact_1.download()

        # Test data and label filenames
        test_data_filename = 'zoo_dataset_X_test'
        test_labels_filename = 'zoo_labels_y_test'
        loaded_data = torch.load(f'{local_dataset_path}/{test_data_filename}')
    
        # Load dataset and labels into notebook
        loaded_labels = torch.load(f'{local_dataset_path}/{test_labels_filename}')
    return loaded_data, loaded_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpret model predictions

    The model returns seven scores for each animal. Taking `argmax` selects a
    numeric class ID. The display dictionary should map those IDs to the same
    classes used by the dataset.

    **The display dictionary below needs correction before interpreting class
    names.** After subtracting one from the UCI labels, ID 0 is mammal and ID 1
    is bird. ID 3 includes fish, and ID 6 covers the dataset’s remaining
    invertebrates, not only crustaceans. Check the
    [UCI class definitions](https://www.archive.ics.uci.edu/dataset/111/zoo)
    when updating that mapping. Numeric prediction comparisons remain the basis
    for accuracy.
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
    Build a `DataFrame` that pairs each prediction with its reference label.
    Use the numeric IDs to check correctness, and verify the display dictionary
    before using class names to explain individual errors.
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
    Count correct and incorrect predictions to see how many test examples
    contribute to the accuracy percentage.
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
    Your results may vary. We found that the model tends to accurately prediect ~80% of the time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verify the Registry collections and lineage

    Navigate to the Registry, and open the
    `zoo-dataset-tensors` and `zoo-dataset-tensors-split` collections. Verify the linked
    versions and four train/test tensor files are there. Then check that the collection we made for our model exists and possesses the model we linked.

    As an exercise, explore the linked model's lineage. To do so, navigate to the registry and collection. Select a model version, then select the **Lineage** tab.
    """)
    return


if __name__ == "__main__":
    app.run()
