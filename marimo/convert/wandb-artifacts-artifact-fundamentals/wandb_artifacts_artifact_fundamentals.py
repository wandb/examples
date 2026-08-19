# /// script
# dependencies = ["wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifact_fundamentals.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{artifacts-fundamentals} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{artifacts-fundamentals} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use [Weights & Biases](https://wandb.com) for machine learning experiment tracking, dataset and model versioning and management, collaboration and more.

    <div><img /></div>

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

    <div><img /></div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use W&B Artifacts to track and version data as the inputs and outputs of your W&B Runs. In addition to logging hyperparameters, metadata, and metrics to a run, you can use an artifact to log the dataset used to train the model as input and the resulting model checkpoints as outputs.

    ![Artifact Simple Diagram](https://docs.wandb.ai/assets/images/artifacts_landing_page2-05443aa39ae53cede7b08908688b334a.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Set Up
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to use Weights & Biases, you will need the `wandb` package installed. You can install it as follows within Colab.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once it is installed, the next step is to import it into your script or notebook with `import wandb`.
    """)
    return


@app.cell
def _():
    import wandb

    return (wandb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also need to authenticate to the Weights & Biases server. There are various ways of doing this, including for [remote or non-interactice workflows](https://docs.wandb.ai/guides/track/environment-variables), but given this is running interactively, we can use `wandb.login()`.

    If we are not already authenticated, a link will appear which you can use to do so.
    """)
    return


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create a Dataset
    Let's create some datasets that we can work with in this example.
    """)
    return


@app.cell
def _():
    import os
    import numpy as np
    import csv

    directory = "dataset"
    os.makedirs(directory, exist_ok=True)
    file1, file2 = os.path.join(directory, "file1.csv"), os.path.join(directory, "file2.csv")

    def generate_dummy_data(num_samples):
        data = [
            np.random.normal(50, 10, num_samples),
            np.random.randint(1, 100, num_samples),
            np.random.choice(['A', 'B', 'C', 'D'], num_samples),
            np.random.uniform(0.0, 1.0, num_samples)
        ]
        return zip(*data)

    def save_to_csv(file, data):
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['feature1', 'feature2', 'feature3', 'feature4'])
            writer.writerows(data)

    num_samples = 100
    save_to_csv(file1, generate_dummy_data(num_samples))
    save_to_csv(file2, generate_dummy_data(num_samples))
    return (directory,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create An Artifact
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The general workflow for creating an Artifact is:

    1.   Intialize a run.
    2.   Create an Artifact.
    3.   Add a any files or directories to the new Artifact that you want to track and version.
    4.   Log the artifact in the W&B platform.

    The most straightforward way of accomplishing this is the second line of code in the example below, which will log, track and version a new dataset (i.e. do points 2, 3, and 4 above in one step).
    """)
    return


@app.cell
def _(directory, wandb):
    _run = wandb.init(project='artifact-basics')
    _run.log_artifact(artifact_or_path=f'{directory}/file1.csv', name='my_first_artifact', type='dataset')
    _run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the above example we first initalize a run using [`wandb.init()`](https://docs.wandb.ai/ref/python/init) the `artifact-basics` project. If this project doesn't exist, it will be created. If it alreadt exists, a new W&B Run will be added to it.

    In the second line we actually log the Artifact with [`run.log_artifact()`](https://docs.wandb.ai/ref/python/public-api/run#log_artifact). In this example, we use three common arguments to the function.
    1. With `artifact_or_path` we specifiy the path to where the data we want to version exists. Any file or directory can be added here.
    2. with `name` we give the artifact a name within Weights & Biases that we will use to access it.
    3. With `type` we give the artifact a higher level grouping. For example, we may have multiple artifacts of type data, and multiple artifacts of type model.

    See the [Artifacts Reference](https://docs.wandb.ai/ref/python/artifact) guide for more information and other commonly used arguments, including how to store additional metadata.

    Each time the above `log_artifact` is executed, wandb will create a new version of the Artifact within Weights & Biases if the underlying data has changed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An alternative approach that offers more control (at the expense of more lines of code) can be seen below.
    """)
    return


@app.cell
def _(directory, wandb):
    _run = wandb.init(project='artifact-basics')
    _artifact = wandb.Artifact('my_first_artifact', type='dataset')
    _artifact.add_file(local_path=f'{directory}/file1.csv')
    # the below will add two individual files to the artifact.
    _artifact.add_file(local_path=f'{directory}/file2.csv')
    _artifact.add_dir(local_path=f'{directory}')
    # or the below if you wanted to add the entire directory contents.
    _run.log_artifact(_artifact)
    # explictly log the artifact to Weights & Biases.
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the above example, lines 3-5 will create a new Artifact within your Weights & Biases project. With the resulting artifact object, you can call the [`artifact.add_file`](https://docs.wandb.ai/ref/python/artifact#add_file) or [`artifact.add_dir`](https://docs.wandb.ai/ref/python/artifact#add_dir) functions in order to add as many files and directories to the Artifact as you want. Once added, the artifact must then be explictly logged to Weights & Biases.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Use an Artifact
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When you want to use a specific version of an Artifact in a downstream task, you can specify the specific version you would like to use via either `v0`, `v1`, `v2` and so on, or via specific aliases you may have added. The `latest` alias always refers to the most recent version of the Artifact logged.

    The proceeding code snippet specifies that the W&B Run will use an artifact called `my_first_artifact` with the alias `latest`:
    """)
    return


@app.cell
def _(wandb):
    _run = wandb.init(project='artifact-basics')
    _artifact = _run.use_artifact(artifact_or_name='my_first_artifact:latest')  # this creates a reference within Weights & Biases that this artifact was used by this run.
    path = _artifact.download()  # this downloads the artifact from Weights & Biases to your local system where the code is executing.
    print(f'Data directory located at {path}')
    _run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For more information on ways to customize your Artifact download, including via the command line, see the [Download and Usage guide](https://docs.wandb.ai/guides/artifacts/download-and-use-an-artifact).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create a new Artifact version
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's say we want to modify our dataset while also tracking and versioning these changes. In the below example we will subsample our dataset and save it as a new file. We will use the [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) library to read our CSV file.

    In the second block of code we will log it to Weights & Biases under the same Artifact name (*my_first_artifact*) so that Weights & Biases knows that this is a new version of an existing artifact.
    """)
    return


@app.cell
def _(directory):
    import pandas
    df = pandas.read_csv(f"{directory}/file1.csv")
    # subsample to 50% of the original size
    df_subsampled = df.sample(frac=0.5, random_state=1)
    # save the subsampled dataframe to a new file.
    df_subsampled.to_csv(f"{directory}/file1.csv", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we have a new subsampled version of our dataset locally, we can log the new version to Weights & Biases.
    """)
    return


@app.cell
def _(directory, wandb):
    _run = wandb.init(project='artifact-basics')
    _run.log_artifact(artifact_or_path=f'{directory}/file1.csv', name='my_first_artifact', type='dataset', aliases=['subsampled'])
    _run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the sampled dataset will be logged to the `my_first_artifact` Artifact as a new version.

    The Artifact has also been given a custom `alias`, which is a unique label for this Artifact version. While the `alias` is currently `subsampled`, the default aliases is `vN`, where `N` is the number of versions the Artifact has. This increments automatically. You can always access specific versions of an Artifact by using an alias.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Update Artifact version metadata
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can update the `description`, `metadata`, and `alias` of an artifact on the W&B platform during or outside a W&B Run.

    This example changes the `description` of the `my_first_artifact` artifact inside a run:
    """)
    return


@app.cell
def _(wandb):
    _run = wandb.init(project='artifact-basics')
    _artifact = _run.use_artifact(artifact_or_name='my_first_artifact:subsampled')
    _artifact.description = 'This is an edited description.'
    _artifact.metadata = {'source': 'local disk', 'internal data owner': 'platform team'}
    _artifact.save()  # persists changes to an Artifact's properties
    _run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Use the Artifact within your pipelines
    Once the artifact is tracked and versioned within Weights & Biases it's now easy to integrate it into your ML workflows.
    """)
    return


@app.cell
def _(wandb):
    _run = wandb.init(project='artifact-basics')
    _artifact = _run.use_artifact(artifact_or_name='my_first_artifact:latest')
    # the below is left as an exercise to the reader
    # train model
    # log model as artifact
    _run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Navigate the Artifacts UI
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also manage your Artifacts via the W&B platform. This can give you insight into your model's performance or dataset versioning. To navigate to the relevant information, click this [link](https://wandb.ai/wandb/artifact-basics/overview), then click on the **Artifacts** tab.

    Navigating to the **Lineage** section in the tab will show the dependency graph formed by calling `run.use_artifact()` when an Artifact is an input to a run, and `run.log_artifact()` when an Artifact is output to a run. This helps visualize the relationship between different model versions and other objects like datasets and jobs in your project. Click [this](https://wandb.ai/wandb/artifact-basics/artifacts/dataset/my_first_artifact/v0/lineage) link to navigate to the project's lineage page.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Naturally, as you integrate W&B Artifacts into your workflow, lineage graphs such as [this interactive example](https://wandb.ai/wandb-smle/artifact_workflow/artifacts/model/quant_model/v16/lineage) will be built up over time, giving you reproducibility, governance, and auditability.

    ![Artifact Lineage Example](https://docs.wandb.ai/assets/images/lineage2a-e3fe54c8916c90499aaf3e1e289062bb.gif)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Next steps
    1. [Artifacts Python reference documentation](https://docs.wandb.ai/ref/python/artifact): Deep dive into artifact parameters and advanced methods.
    2. [Lineage](https://docs.wandb.ai/guides/artifacts/explore-and-traverse-an-artifact-graph): View lineage graphs, which are automatically built when using W&B artifact system, providing an auditable visual overview of the relationships between specific artifact versions, datasets models and runs.
    3. [Model Registry](https://docs.wandb.ai/guides/model_registry): Learn how to centralize your best artifact versions in a shared registry.
    4. [Artifact Automations](https://docs.wandb.ai/guides/artifacts/project-scoped-automations): Automatically run specific Weights & Biases jobs based on changes to your artifacts, such as automatically training a new model each time a new version of the training data is logged.
    5. [Reference Artifacts](https://docs.wandb.ai/guides/artifacts/track-external-files#download-a-reference-artifact): Track files saved outside the W&B server, like Amazon S3 buckets, GCS buckets, Azure blobs, and more.
    """)
    return


if __name__ == "__main__":
    app.run()
