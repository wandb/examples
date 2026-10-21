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
    teams. This lesson converts the UCI Zoo dataset to tensors, logs and links
    those files to a Registry collection, trains a classifier, and retrieves the
    saved model and test data for inference.

    The classifier uses animal attributes such as feathers, fins, and number of
    legs to predict one of seven dataset classes. The model is intentionally
    small so the lesson can focus on Artifact paths, version selection, and lineage.

    Before publishing, check that the target registries exist and that your
    account can link versions to their collections. Registry access also determines
    which colleagues can retrieve the linked files.
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
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `zoo_experiment`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="zoo_experiment", label="Project", full_width=True),
    ).form(submit_button_label="Connect to Weights & Biases", bordered=True)
    wandb_setup_form
    return (wandb_setup_form,)


@app.cell(hide_code=True)
def _(mo, wandb, wandb_setup_form):
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _api_key = _submission["api_key"].strip()
    _entity = _submission["entity"].strip()
    _project = _submission["project"].strip()
    mo.stop(
        not _entity or not _project,
        mo.callout("Enter your team and project, then connect again.", kind="warn"),
    )

    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout("Check your API key or saved credentials, then connect again.", kind="danger"),
    )

    TEAM_ENTITY = _entity
    PROJECT = _project
    mo.md("✓ Connected to Weights & Biases.")
    return PROJECT, TEAM_ENTITY


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

    Convert feature values to `float32`, matching the linear layers’ weights.
    Convert labels to integer tensors and subtract one so the original class
    IDs 1–7 become 0–6, as required by the seven-output classifier.

    Keep the numeric class IDs consistent through training and evaluation.
    Human-readable labels are added only when displaying predictions.
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

    An Artifact stores a versioned set of files in a project. A Registry collection
    contains links to Artifact versions so authorized users can discover and reuse
    them. Linking creates a reference to the source version.

    The target path identifies the registry and collection. The registry must
    already exist. This example uses a `dataset` registry and a collection named
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

    Define the target path, start a run, and create a dataset Artifact. Add the
    feature and label tensor files, then call `run.link_artifact()` with the
    Artifact and target path. For this draft Artifact, the call logs the files
    as a run output and links the resulting version to the collection.

    These first files contain the full dataset. The following section creates
    and publishes separate training and test files.
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
    The next cell uses the team and project from the setup form. It adds
    `zoo_dataset.pt` and `zoo_labels.pt` to a dataset Artifact, links the version
    to `dataset_target_path`, and finishes the run.
    """)
    return


@app.cell
def _(PROJECT, TEAM_ENTITY, dataset_target_path, wandb):

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
    We can verify we correctly linked our artifact to our desired collection and registry with Weights & Biases App UI:

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
    ## Define a classifier

    The network maps 16 animal attributes to 16 hidden units with a sigmoid
    activation, then produces seven class scores. Cross-entropy loss accepts
    these scores directly. The same architecture must be recreated when loading
    the saved state dictionary later.
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
    ## Train from a registered dataset

    Retrieve the split dataset through its Registry path so the training run
    records exactly which Artifact version it consumed.

    ```python
    artifact_name = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"
    ```

    The code below selects `v0`. That is appropriate only if it is the split you
    intend to train on. If the collection already existed or you reran publication,
    inspect its versions and update `VERSION` before training. A Registry collection’s
    version number need not match the source project’s version number.

    A target path identifies the collection used when linking. Adding a version
    suffix selects a specific entry to download.
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
    The training run downloads the feature and label files from the registered
    split, updates the model, and logs training loss. Every 100 iterations,
    including iteration zero, it saves weights when the loss improves over the
    previous saved checkpoint.

    The checkpoint file is overwritten locally. After training, the code uploads
    only the final saved checkpoint as one model Artifact version. It does not
    upload a version at every checkpoint interval, and checkpoint selection uses
    training loss rather than held-out performance.

    The model is now stored in the selected project. The next step links that
    version to Registry, where access is governed by the registry’s permissions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Link the trained model to Registry

    First identify the model version stored in the team project. Its path contains
    the team, project, Artifact name, and version suffix.

    ```python
    artifact_name = f"{TEAM_ENTITY}/{PROJECT}/{ARTIFACT_NAME}:v{VERSION}"
    ```

    This example uses a run-specific model name and selects its first version,
    `v0`. If you adapt the code to reuse Artifact names, choose the intended
    version explicitly before linking it.
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
    Define the destination collection for the model version. The next cell uses
    the `model` registry and the `Zoo_Classifier_Models` collection. Check that
    the registry name matches your organization before running the linking cell.

    ```python
    target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
    ```

    The source Artifact path selects the files to share. The target path selects
    where users will discover that version in Registry.
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
    ### Inspect the registered model’s lineage

    Open the model version in Registry and inspect its **Lineage** view. Follow
    the input edge to the split dataset and the output edge from the training
    run to the model. This connects the saved weights to the data version used
    for training.

    The screenshot below is an earlier example. Its run names and version numbers
    will differ from yours.

    ![Example dataset and model lineage](./images/dag_model_registry.png)

    Use the lineage graph to choose a model version based on its recorded inputs
    and training run, then copy that version’s full path for inference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download registered artifacts for inference

    The final section simulates using the registered model from another team’s
    project. It downloads model weights and the test split, recreates the model,
    and compares predictions with the saved labels.

    The API key submitted earlier must have access to the Registry versions and
    permission to create runs in the destination project. Changing the destination
    team does not authenticate as a different person.

    Open each version in Registry and copy its full Artifact path, or construct
    the path from the registry, collection, and version. The examples below use
    `VERSION_1 = 0` and `VERSION_2 = 0`. Update them if the intended model or test
    split has another version number.
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
    Replace `<TEAM_B>` with a team where your authenticated account can create
    runs. You can reuse the team selected in the setup form if you do not have
    a second team. The run records the downloaded model as an input.
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
    Build a DataFrame that pairs each prediction with its reference label.
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
    Use the percentage calculated from your current predictions. The historical
    output below reports about 88%, but it is not a target or a result produced
    by the current run. This small test split can give noticeably different
    scores after changes to training or initialization.
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
    ## Review the saved experiment

    You have logged dataset files, linked their versions to Registry, trained
    from a registered split, and retrieved a model and test data for inference.
    The Registry paths and lineage connect these steps across projects.

    Inspect incorrect predictions alongside the original animal attributes.
    The test split is small, so avoid drawing broad conclusions from one accuracy
    value. If you use these examples to change the model, reserve other data for
    a final evaluation.

    ## Document a model version for reuse

    Add a description explaining the expected input features, network architecture,
    class-ID mapping, and how to load the state dictionary. Include the training
    data version and evaluation results. This gives another user the information
    needed to use the saved model correctly.
    """)
    return


if __name__ == "__main__":
    app.run()
