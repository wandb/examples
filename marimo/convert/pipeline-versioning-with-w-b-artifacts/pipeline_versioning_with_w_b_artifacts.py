# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24.0",
#     "torch>=2.6",
#     "torchvision>=0.21",
#     "wandb>=0.19.10",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    app_title="Pipeline Versioning with W&B Artifacts",
)


@app.cell
def _():
    import os
    import random
    import tempfile

    import marimo as mo
    import torch
    import torchvision
    import wandb
    from torch.utils.data import TensorDataset

    return TensorDataset, mo, os, random, tempfile, torch, torchvision, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/pipeline-versioning-with-w-b-artifacts/pipeline_versioning_with_w_b_artifacts.py/server)

    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Artifacts

    In this notebook, we'll show you how to use W&B Artifacts (🏺)
    to track your ML experiment pipelines (🧪).
    Our sophisticated mathematical models predict the following result:

    $$
    {\Huge
    🧪 + 🏺 = 😃}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Follow along with the video tutorial

    Watch the tutorial below, or [open it on YouTube](https://youtu.be/Hd94gatGMic).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(r"""
    <iframe
      width="100%"
      height="450"
      src="https://www.youtube.com/embed/Hd94gatGMic?rel=0"
      title="Pipeline Versioning with W&B Artifacts"
      frameborder="0"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
      referrerpolicy="strict-origin-when-cross-origin"
      allowfullscreen>
    </iframe>
    <p><a href="https://youtu.be/Hd94gatGMic" target="_blank" rel="noopener noreferrer">Open the video on YouTube</a></p>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What are Artifacts and Why Should I Care?

    An "artifact", like a Greek [amphora 🏺](https://en.wikipedia.org/wiki/Amphora),
    is a produced object -- the output of a process.
    In ML, the most important artifacts are _datasets_ and _models_.

    And, like the [Cross of Coronado](https://indianajones.fandom.com/wiki/Cross_of_Coronado), these important artifacts belong in a museum!
    That is, they should be cataloged and organized
    so that you, your team, and the ML community at large can learn from them.
    After all, those who don't track training are doomed to repeat it.

    Using our Artifacts API, you can log `Artifact`s as outputs of W&B `Run`s or use `Artifact`s as input to `Run`s, as in this diagram,
    where a training run takes in a dataset and produces a model.

     ![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M94QAXA-oJmE6q07_iT%2F-M94QJCXLeePzH1p_fW1%2Fsimple%20artifact%20diagram%202.png?alt=media&token=94bc438a-bd3b-414d-a4e4-aa4f6f359f21)

    Since one run can use another's output as an input, Artifacts and Runs together form a directed graph -- actually, a bipartite [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)! -- with nodes for `Artifact`s and `Run`s
    and arrows connecting `Run`s to the `Artifact`s they consume or produce.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The notebook metadata declares W&B, PyTorch, and torchvision. The visible
    import cell above is the complete shared runtime setup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Authentication

    Submit the form before running the pipeline. Leave the API key blank to
    use `WANDB_API_KEY` from molab's Secrets panel or credentials already
    configured in this runtime. Editing an unsubmitted field does not log in,
    download MNIST, or create a W&B run.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _wandb_entity = mo.ui.text(
        value="",
        label="W&B entity -- a team you belong to (blank uses your default)",
        full_width=True,
    )
    _wandb_project = mo.ui.text(
        value="artifacts-example",
        label="W&B project",
        full_width=True,
    )
    _wandb_api_key = mo.ui.text(
        value="",
        kind="password",
        label="W&B API key (blank uses runtime credentials)",
        full_width=True,
    )
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}\n\n{project}")
        .batch(
            api_key=_wandb_api_key,
            entity=_wandb_entity,
            project=_wandb_project,
        )
        .form(submit_button_label="Connect to W&B", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(mo, wandb, wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            mo.md("Submit the authentication form to continue."),
            kind="info",
        ),
    )
    _login_values = wandb_login_form.value
    _api_key = _login_values["api_key"].strip()
    try:
        _login_ok = wandb.login(
            key=_api_key or None,
            relogin=bool(_api_key),
        )
    except (wandb.errors.Error, ValueError):
        _login_ok = False
    mo.stop(
        not _login_ok,
        mo.callout(
            mo.md(
                "W&B authentication did not complete. Check the API key or "
                "configure `WANDB_API_KEY` in molab's Secrets panel, then submit again."
            ),
            kind="danger",
        ),
    )
    wandb_session = {
        "entity": _login_values["entity"].strip() or None,
        "project": _login_values["project"].strip() or "artifacts-example",
    }
    mo.callout(
        mo.md(
            f"Connected. Pipeline runs will be written to "
            f"**{wandb_session['project']}** only after you click a step button."
        ),
        kind="success",
    )
    return (wandb_session,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Log a Dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, let's define some Artifacts.

    This example is based off of this PyTorch
    ["Basic MNIST Example"](https://github.com/pytorch/examples/tree/master/mnist/),
    but could just as easily have been done in TensorFlow, in any other framework,
    or in pure Python.

    We start with the `Dataset`s:
    - a `train`ing set, for choosing the parameters,
    - a `validation` set, for choosing the hyperparameters,
    - a `test`ing set, for evaluating the final model

    The first cell below defines these three datasets.
    """)
    return


@app.cell
def _(random, torch, torchvision):
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Data parameters
    num_classes = 10
    input_shape = (1, 28, 28)

    # drop slow mirror from list of MNIST mirrors
    torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                          if not mirror.startswith("http://yann.lecun.com")]
    return device, input_shape, num_classes


@app.cell
def _(TensorDataset, torchvision):
    def load(data_root, train_size=50_000):
        """
        # Load the data
        """

        # the data, split between train and test sets
        train = torchvision.datasets.MNIST(data_root, train=True, download=True)
        test = torchvision.datasets.MNIST(data_root, train=False, download=True)
        (x_train, y_train), (x_test, y_test) = (
            (train.data, train.targets),
            (test.data, test.targets),
        )

        # split off a validation set for hyperparameter tuning
        x_train, x_val = x_train[:train_size], x_train[train_size:]
        y_train, y_val = y_train[:train_size], y_train[train_size:]

        training_set = TensorDataset(x_train, y_train)
        validation_set = TensorDataset(x_val, y_val)
        test_set = TensorDataset(x_test, y_test)

        datasets = [training_set, validation_set, test_set]

        return datasets

    return (load,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This sets up a pattern we'll see repeated in this example:
    the code to log the data as an Artifact is wrapped around the code for
    producing that data.
    In this case, the code for `load`ing the data is
    separated out from the code for `load_and_log`ging the data.

    This is good practice!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to log these datasets as Artifacts,
    we just need to
    1. create a `Run` with `wandb.init`,
    2. create an `Artifact` for the dataset, and
    3. save and log the associated `file`s with `run.log_artifact`.

    Check out the example code below, then click **Log raw dataset** to run
    this pipeline step.
    """)
    return


@app.cell
def _(load, tempfile, torch, wandb):
    def load_and_log(session):
        workspace_context = tempfile.TemporaryDirectory(
            prefix="wandb-artifacts-load-"
        )
        workspace = workspace_context.name
        # 🚀 start a run, with a type to label it and a project it can call home
        with wandb.init(
            project=session["project"],
            entity=session["entity"],
            job_type="load-data",
            reinit="create_new",
            dir=workspace,
        ) as run:
            datasets = load(workspace)  # separate code for loading the datasets
            names = ["training", "validation", "test"]

            # 🏺 create our Artifact
            raw_data = wandb.Artifact(
                "mnist-raw", type="dataset",
                description="Raw MNIST dataset, split into train/val/test",
                metadata={
                    "source": "torchvision.datasets.MNIST",
                    "sizes": [len(dataset) for dataset in datasets],
                },
            )

            for name, data in zip(names, datasets):
                # 🐣 Store a new file in the artifact, and write something into its contents.
                with raw_data.new_file(name + ".pt", mode="wb") as file:
                    x, y = data.tensors
                    torch.save((x, y), file)

            # ✍️ Save the artifact to W&B.
            logged_artifact = run.log_artifact(raw_data).wait()
            result = {
                "artifact_path": logged_artifact.qualified_name,
                "run_url": run.url,
            }

        workspace_context.cleanup()
        return result

    return (load_and_log,)


@app.cell(hide_code=True)
def _(mo):
    log_raw_data = mo.ui.run_button(label="Log raw dataset")
    log_raw_data
    return (log_raw_data,)


@app.cell
def _(load_and_log, log_raw_data, mo, wandb_session):
    mo.stop(
        not log_raw_data.value,
        mo.md("Click **Log raw dataset** to create the first W&B Artifact."),
    )
    raw_data_result = load_and_log(wandb_session)
    mo.callout(
        mo.md(
            "Raw MNIST Artifact logged. "
            f"[Open the W&B run]({raw_data_result['run_url']})."
        ),
        kind="success",
    )
    return (raw_data_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `wandb.init`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When we make the `Run` that's going to produce the `Artifact`s,
    we need to state which `project` it belongs to.

    Depending on your workflow,
    a project might be as big as `car-that-drives-itself`
    or as small as `iterative-architecture-experiment-117`.

    > **Rule of 👍**: if you can, keep all of the `Run`s that share `Artifact`s
    inside a single project. This keeps things simple,
    but don't worry -- `Artifact`s are portable across projects!

    To help keep track of all the different kinds of jobs you might run,
    it's useful to provide a `job_type` when making `Runs`.
    This keeps the graph of your Artifacts nice and tidy.

    > **Rule of 👍**: the `job_type` should be descriptive and correspond to a single step of your pipeline. Here, we separate out `load`ing data from `preprocess`ing data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `wandb.Artifact`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To log something as an `Artifact`, we have to first make an `Artifact` object.

    Every `Artifact` has a `name` -- that's what the first argument sets.

    > **Rule of 👍**: the `name` should be descriptive, but easy to remember and type --
    we like to use names that are hyphen-separated and correspond to variable names in the code.

    It also has a `type`. Just like `job_type`s for `Run`s,
    this is used for organizing the graph of `Run`s and `Artifact`s.

    > **Rule of 👍**: the `type` should be simple:
    more like `dataset` or `model`
    than `mnist-data-YYYYMMDD`.

    You can also attach a `description` and some `metadata`, as a dictionary.
    The `metadata` just needs to be serializable to JSON.

    > **Rule of 👍**: the `metadata` should be as descriptive as possible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `artifact.new_file` and `run.log_artifact`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once we've made an `Artifact` object, we need to add files to it.

    You read that right: _files_ with an _s_.
    `Artifact`s are structured like directories,
    with files and sub-directories.

    > **Rule of 👍**: whenever it makes sense to do so, split the contents
    of an `Artifact` up into multiple files. This will help if it comes time to scale!

    We use the `new_file` method
    to simultaneously write the file and attach it to the `Artifact`.
    Below, we'll use the `add_file` method,
    which separates those two steps.

    Once we've added all of our files, we need to `log_artifact` to [wandb.ai](https://wandb.ai).

    You'll notice some URLs appeared in the output,
    including one for the Run page.
    That's where you can view the results of the `Run`,
    including any `Artifact`s that got logged.

    We'll see some examples that make better use of the other components of the Run page below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Use a Logged Dataset Artifact
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Artifact`s in W&B, unlike artifacts in museums,
    are designed to be _used_, not just stored.

    Let's see what that looks like.

    The cell below defines a pipeline step that takes in a raw dataset
    and uses it to produce a `preprocess`ed dataset:
    `normalize`d and shaped correctly.

    Notice again that we split out the meat of the code, `preprocess`,
    from the code that interfaces with `wandb`.
    """)
    return


@app.cell
def _(TensorDataset, torch):
    def preprocess(dataset, normalize=True, expand_dims=True):
        """
        ## Prepare the data
        """
        x, y = dataset.tensors

        if normalize:
            # Scale images to the [0, 1] range
            x = x.type(torch.float32) / 255

        if expand_dims:
            # Make sure images have shape (1, 28, 28)
            x = torch.unsqueeze(x, 1)

        return TensorDataset(x, y)

    return (preprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now for the code that instruments this `preprocess` step with `wandb.Artifact` logging.

    Note that the example below both `use`s an `Artifact`,
    which is new,
    and `log`s it,
    which is the same as the last step.
    `Artifact`s are both the inputs and the outputs of `Run`s!

    We use a new `job_type`, `preprocess-data`,
    to make it clear that this is a different kind of job from the previous one.
    """)
    return


@app.cell
def _(TensorDataset, os, torch):
    def read(data_dir, split):
        filename = split + ".pt"
        x, y = torch.load(
            os.path.join(data_dir, filename),
            weights_only=True,
        )

        return TensorDataset(x, y)

    return (read,)


@app.cell
def _(os, preprocess, read, tempfile, torch, wandb):
    def preprocess_and_log(raw_artifact_path, session, steps):
        workspace_context = tempfile.TemporaryDirectory(
            prefix="wandb-artifacts-preprocess-"
        )
        workspace = workspace_context.name
        with wandb.init(
            project=session["project"],
            entity=session["entity"],
            job_type="preprocess-data",
            reinit="create_new",
            dir=workspace,
        ) as run:
            processed_data = wandb.Artifact(
                "mnist-preprocess",
                type="dataset",
                description="Preprocessed MNIST dataset",
                metadata=steps,
            )

            # ✔️ declare which artifact we'll be using
            raw_data_artifact = run.use_artifact(raw_artifact_path, type="dataset")

            # 📥 if need be, download the artifact
            raw_dataset = raw_data_artifact.download(
                root=os.path.join(workspace, "raw-data")
            )

            for split in ["training", "validation", "test"]:
                raw_split = read(raw_dataset, split)
                processed_dataset = preprocess(raw_split, **steps)

                with processed_data.new_file(split + ".pt", mode="wb") as file:
                    x, y = processed_dataset.tensors
                    torch.save((x, y), file)

            logged_artifact = run.log_artifact(processed_data).wait()
            result = {
                "artifact_path": logged_artifact.qualified_name,
                "run_url": run.url,
            }

        workspace_context.cleanup()
        return result

    return (preprocess_and_log,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One thing to notice here is that the `steps` of the preprocessing
    are saved with the `preprocessed_data` as `metadata`.

    If you're trying to make your experiments reproducible,
    capturing lots of metadata is a good idea!

    Also, even though our dataset is a "`large artifact`",
    the `download` step is done in much less than a second.

    Expand the markdown cell below for details.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    preprocess_data = mo.ui.run_button(label="Preprocess dataset")
    preprocess_data
    return (preprocess_data,)


@app.cell
def _(mo, preprocess_and_log, preprocess_data, raw_data_result, wandb_session):
    mo.stop(
        not preprocess_data.value,
        mo.md("Click **Preprocess dataset** after logging the raw dataset."),
    )
    steps = {"normalize": True,
             "expand_dims": True}

    preprocess_result = preprocess_and_log(
        raw_data_result["artifact_path"],
        wandb_session,
        steps,
    )
    mo.callout(
        mo.md(
            "Preprocessed MNIST Artifact logged. "
            f"[Open the W&B run]({preprocess_result['run_url']})."
        ),
        kind="success",
    )
    return (preprocess_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `run.use_artifact`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These steps are simpler. The consumer just needs to know the `name` of the `Artifact`, plus a bit more.

    That "bit more" is the `alias` of the particular version of the `Artifact` you want.

    By default, the last version to be uploaded is tagged `latest`.
    Otherwise, you can pick older versions with `v0`/`v1`, etc.,
    or you can provide your own aliases, like `best` or `jit-script`.
    Just like [Docker Hub](https://hub.docker.com/) tags,
    aliases are separated from names with `:`,
    so the `Artifact` we want is `mnist-raw:latest`.

    > **Rule of 👍**: Keep aliases short and sweet.
    Use custom `alias`es like `latest` or `best` when you want an `Artifact`
    that satisfies some property
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `artifact.download`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, you may be worrying about the `download` call.
    If we download another copy, won't that double the burden on memory?

    Don't worry friend. Before we actually download anything,
    we check to see if the right version is available locally.
    This uses the same technology that underlies [torrenting](https://en.wikipedia.org/wiki/Torrent_file) and [version control with `git`](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html): hashing.

    `artifact.download()` materializes a version under the `root` you provide
    and reuses W&B's content-addressed cache when possible. This notebook passes
    an isolated temporary root to every pipeline stage, then removes it after
    the run and Artifact upload finish. That keeps downloaded tensors and W&B
    runtime files out of the repository.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In your own workflow, pass a persistent `root=` to `artifact.download()` if
    you want to inspect the materialized files after the stage completes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Artifacts page on [wandb.ai](https://wandb.ai)

    Now that we've logged and used an `Artifact`,
    let's check out the Artifacts tab on the Run page.

    Navigate to the Run page URL from the `wandb` output
    and select the "Artifacts" tab from the left sidebar
    (it's the one with the database icon,
    which looks like three hockey pucks stacked on top of one another).

    Click a row in either the "Input Artifacts" table
    or in the "Output Artifacts" table,
    then check out the tabs ("Overview", "Metadata")
    to see everything logged about the `Artifact`.

    We particularly like the "Graph View".
    By default, it shows a graph
    with the `type`s of `Artifact`s
    and the `job_type`s of `Run` as the two types of nodes,
    with arrows to represent consumption and production.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Log a Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That's enough to see how the API for `Artifact`s works,
    but let's follow this example through to the end of the pipeline
    so we can see how `Artifact`s can improve your ML workflow.

    This first cell here builds a DNN `model` in PyTorch -- a really simple ConvNet.

    We'll start by just initializing the `model`, not training it.
    That way, we can repeat the training while keeping everything else constant.
    """)
    return


@app.cell
def _(input_shape, num_classes):
    from math import floor

    import torch.nn as nn

    class ConvNet(nn.Module):
        def __init__(self, hidden_layer_sizes=[32, 64],
                      kernel_sizes=[3],
                      activation="ReLU",
                      pool_sizes=[2],
                      dropout=0.5,
                      num_classes=num_classes,
                      input_shape=input_shape):
  
            super(ConvNet, self).__init__()

            self.layer1 = nn.Sequential(
                  nn.Conv2d(in_channels=input_shape[0], out_channels=hidden_layer_sizes[0], kernel_size=kernel_sizes[0]),
                  getattr(nn, activation)(),
                  nn.MaxPool2d(kernel_size=pool_sizes[0])
            )
            self.layer2 = nn.Sequential(
                  nn.Conv2d(in_channels=hidden_layer_sizes[0], out_channels=hidden_layer_sizes[-1], kernel_size=kernel_sizes[-1]),
                  getattr(nn, activation)(),
                  nn.MaxPool2d(kernel_size=pool_sizes[-1])
            )
            self.layer3 = nn.Sequential(
                  nn.Flatten(),
                  nn.Dropout(dropout)
            )

            fc_input_dims = floor((input_shape[1] - kernel_sizes[0] + 1) / pool_sizes[0]) # layer 1 output size
            fc_input_dims = floor((fc_input_dims - kernel_sizes[-1] + 1) / pool_sizes[-1]) # layer 2 output size
            fc_input_dims = fc_input_dims*fc_input_dims*hidden_layer_sizes[-1] # layer 3 output size

            self.fc = nn.Linear(fc_input_dims, num_classes)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.fc(x)
            return x

    return (ConvNet,)


@app.cell
def _(ConvNet):
    model_preview = ConvNet()
    model_preview
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we're using W&B to track the run,
    and so using the [`wandb.config`](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/configs-in-w-b/configs_in_w_b.py/server)
    object to store all of the hyperparameters.

    The `dict`ionary version of that `config` object is a really useful piece of `metadata`, so make sure to include it!
    """)
    return


@app.cell
def _(ConvNet, os, tempfile, torch, wandb):
    def build_model_and_log(config, session):
        workspace_context = tempfile.TemporaryDirectory(
            prefix="wandb-artifacts-initialize-"
        )
        workspace = workspace_context.name
        with wandb.init(
            project=session["project"],
            entity=session["entity"],
            job_type="initialize",
            config=config,
            reinit="create_new",
            dir=workspace,
        ) as run:
            run_config = run.config

            model = ConvNet(**run_config)

            model_artifact = wandb.Artifact(
                "convnet",
                type="model",
                description="Simple AlexNet style CNN",
                metadata=dict(run_config),
            )

            checkpoint_path = os.path.join(workspace, "initialized_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            # ➕ another way to add a file to an Artifact
            model_artifact.add_file(checkpoint_path, name="initialized_model.pth")

            logged_artifact = run.log_artifact(model_artifact).wait()
            result = {
                "artifact_path": logged_artifact.qualified_name,
                "run_url": run.url,
            }

        workspace_context.cleanup()
        return result

    return (build_model_and_log,)


@app.cell(hide_code=True)
def _(mo):
    initialize_model = mo.ui.run_button(label="Initialize model")
    initialize_model
    return (initialize_model,)


@app.cell
def _(
    build_model_and_log,
    initialize_model,
    mo,
    preprocess_result,
    wandb_session,
):
    mo.stop(
        not initialize_model.value,
        mo.md("Click **Initialize model** after preprocessing the dataset."),
    )
    model_config = {"hidden_layer_sizes": [32, 64],
                    "kernel_sizes": [3],
                    "activation": "ReLU",
                    "pool_sizes": [2],
                    "dropout": 0.5,
                    "num_classes": 10}

    _ = preprocess_result
    initialized_model_result = build_model_and_log(
        model_config,
        wandb_session,
    )
    mo.callout(
        mo.md(
            "Initialized model Artifact logged. "
            f"[Open the W&B run]({initialized_model_result['run_url']})."
        ),
        kind="success",
    )
    return (initialized_model_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `artifact.add_file`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Instead of simultaneously writing a `new_file` and adding it to the `Artifact`,
    as in the dataset logging examples,
    we can also write files in one step
    (here, `torch.save`)
    and then `add` them to the `Artifact` in another.

    > **Rule of 👍**: use `new_file` when you can, to prevent duplication.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Use a Logged Model Artifact
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Just like we could call `use_artifact` on a `dataset`,
    we can call it on our `initialized_model`
    to use it in another `Run`.

    This time, let's `train` the `model`.

    For more details, see the notebook on
    [instrumenting W&B with PyTorch](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/simple-pytorch-integration/simple_pytorch_integration.py/server).
    """)
    return


@app.cell
def _():
    import torch.nn.functional as F

    return (F,)


@app.cell
def _(F, device, evaluate_loader, torch):
    def train(model, train_loader, valid_loader, config, run):
        optimizer = getattr(torch.optim, config.optimizer)(model.parameters())
        example_ct = 0
        for epoch in range(config.epochs):
            # Evaluation switches the model to eval mode, so restore training mode
            # before every epoch (important for the Dropout layer).
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

                example_ct += len(data)

                if batch_idx % config.batch_log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )

                    train_log(run, loss, example_ct, epoch)

            # evaluate the model on the validation set at each epoch
            loss, accuracy = evaluate_loader(model, valid_loader)
            validation_log(run, loss, accuracy, example_ct, epoch)

    return (train,)


@app.cell
def _(F, device, torch):
    def evaluate_loader(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction="sum")
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)

        return test_loss, accuracy

    return (evaluate_loader,)


@app.function
def train_log(run, loss, example_ct, epoch):
    loss = loss.detach().item()

    # where the magic happens
    run.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


@app.function
def validation_log(run, loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # where the magic happens
    run.log(
        {
            "epoch": epoch,
            "validation/loss": loss,
            "validation/accuracy": accuracy,
        },
        step=example_ct,
    )
    print(
        f"Loss/accuracy after "
        + str(example_ct).zfill(5)
        + f" examples: {loss:.3f}/{accuracy:.3f}"
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll run two separate `Artifact`-producing `Run`s this time.

    Once the first finishes `train`ing the `model`,
    the `second` will consume the `trained-model` `Artifact`
    by `evaluate`ing its performance on the `test_dataset`.

    Also, we'll pull out the 32 examples on which the network gets the most confused --
    on which the `categorical_crossentropy` is highest.

    This is a good way to diagnose issues with your dataset and your model!
    """)
    return


@app.cell
def _(evaluate_loader, get_hardest_k_examples):
    def evaluate(model, test_loader):
        """
        ## Evaluate the trained model
        """

        loss, accuracy = evaluate_loader(model, test_loader)
        highest_losses, hardest_examples, true_labels, predictions = (
            get_hardest_k_examples(model, test_loader.dataset)
        )

        return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

    return (evaluate,)


@app.cell
def _(F, device, torch):
    def get_hardest_k_examples(model, testing_set, k=32):
        model.eval()

        loader = torch.utils.data.DataLoader(testing_set, 1, shuffle=False)

        # get the losses and predictions for each item in the dataset
        losses = []
        predictions = []
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.cross_entropy(output, target)
                pred = output.argmax(dim=1, keepdim=True)

                losses.append(loss.detach().cpu())
                predictions.append(pred.detach().cpu().squeeze(1))

        losses = torch.stack(losses)
        predictions = torch.cat(predictions)
        argsort_loss = torch.argsort(losses)
        hardest_indices = argsort_loss[-k:]

        highest_k_losses = losses[hardest_indices]
        hardest_k_examples = testing_set.tensors[0][hardest_indices]
        true_labels = testing_set.tensors[1][hardest_indices]
        predicted_labels = predictions[hardest_indices]

        return highest_k_losses, hardest_k_examples, true_labels, predicted_labels

    return (get_hardest_k_examples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These logging functions don't add any new `Artifact` features,
    so we won't comment on them:
    we're just `use`ing, `download`ing,
    and `log`ging `Artifact`s.
    """)
    return


@app.cell
def _(ConvNet, device, os, read, tempfile, torch, train, wandb):
    def train_and_log(
        config,
        initialized_model_path,
        processed_data_path,
        session,
    ):
        workspace_context = tempfile.TemporaryDirectory(
            prefix="wandb-artifacts-train-"
        )
        workspace = workspace_context.name
        with wandb.init(
            project=session["project"],
            entity=session["entity"],
            job_type="train",
            config=config,
            reinit="create_new",
            dir=workspace,
        ) as run:
            run_config = run.config

            data = run.use_artifact(processed_data_path, type="dataset")
            data_dir = data.download(root=os.path.join(workspace, "data"))

            training_dataset = read(data_dir, "training")
            validation_dataset = read(data_dir, "validation")

            train_loader = torch.utils.data.DataLoader(
                training_dataset,
                batch_size=run_config.batch_size,
            )
            validation_loader = torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=run_config.batch_size,
            )

            model_artifact = run.use_artifact(
                initialized_model_path,
                type="model",
            )
            model_dir = model_artifact.download(
                root=os.path.join(workspace, "initialized-model")
            )
            model_path = os.path.join(model_dir, "initialized_model.pth")
            model_config = model_artifact.metadata
            run_config.update(model_config)

            model = ConvNet(**model_config)
            model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
            model = model.to(device)

            train(model, train_loader, validation_loader, run_config, run)

            trained_model_artifact = wandb.Artifact(
                "trained-model",
                type="model",
                description="Trained NN model",
                metadata=dict(model_config),
            )

            trained_model_path = os.path.join(workspace, "trained_model.pth")
            torch.save(model.state_dict(), trained_model_path)
            trained_model_artifact.add_file(
                trained_model_path,
                name="trained_model.pth",
            )
            logged_model = run.log_artifact(trained_model_artifact).wait()
            result = {
                "artifact_path": logged_model.qualified_name,
                "run_url": run.url,
            }

        workspace_context.cleanup()
        return result

    return (train_and_log,)


@app.cell
def _(ConvNet, device, evaluate, os, read, tempfile, torch, wandb):
    def evaluate_and_log(
        processed_data_path,
        session,
        trained_model_path,
        config=None,
    ):
        workspace_context = tempfile.TemporaryDirectory(
            prefix="wandb-artifacts-evaluate-"
        )
        workspace = workspace_context.name
        with wandb.init(
            project=session["project"],
            entity=session["entity"],
            job_type="report",
            config=config,
            reinit="create_new",
            dir=workspace,
        ) as run:
            data = run.use_artifact(processed_data_path, type="dataset")
            data_dir = data.download(root=os.path.join(workspace, "data"))
            testing_set = read(data_dir, "test")

            test_loader = torch.utils.data.DataLoader(
                testing_set,
                batch_size=128,
                shuffle=False,
            )

            model_artifact = run.use_artifact(trained_model_path, type="model")
            model_dir = model_artifact.download(
                root=os.path.join(workspace, "trained-model")
            )
            model_path = os.path.join(model_dir, "trained_model.pth")
            model_config = model_artifact.metadata

            model = ConvNet(**model_config)
            model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
            model = model.to(device)

            loss, accuracy, highest_losses, hardest_examples, true_labels, preds = (
                evaluate(model, test_loader)
            )

            run.summary.update(
                {"loss": float(loss), "accuracy": float(accuracy)}
            )

            run.log(
                {
                    "high-loss-examples": [
                        wandb.Image(
                            hard_example.detach()
                            .cpu()
                            .clamp(0, 1)
                            .mul(255)
                            .round()
                            .to(torch.uint8),
                            caption=str(int(pred)) + "," + str(int(label)),
                        )
                        for hard_example, pred, label in zip(
                            hardest_examples,
                            preds,
                            true_labels,
                        )
                    ]
                }
            )
            result = {
                "accuracy": float(accuracy),
                "loss": float(loss),
                "run_url": run.url,
            }

        workspace_context.cleanup()
        return result

    return (evaluate_and_log,)


@app.cell(hide_code=True)
def _(mo):
    train_and_evaluate = mo.ui.run_button(label="Train and evaluate")
    train_and_evaluate
    return (train_and_evaluate,)


@app.cell
def _(
    evaluate_and_log,
    initialized_model_result,
    mo,
    preprocess_result,
    train_and_evaluate,
    train_and_log,
    wandb_session,
):
    mo.stop(
        not train_and_evaluate.value,
        mo.md("Click **Train and evaluate** after initializing the model."),
    )
    train_config = {"batch_size": 128,
                    "epochs": 5,
                    "batch_log_interval": 25,
                    "optimizer": "Adam"}

    trained_model_result = train_and_log(
        train_config,
        initialized_model_result["artifact_path"],
        preprocess_result["artifact_path"],
        wandb_session,
    )
    evaluation_result = evaluate_and_log(
        preprocess_result["artifact_path"],
        wandb_session,
        trained_model_result["artifact_path"],
    )
    mo.callout(
        mo.md(
            "Training and evaluation complete. "
            f"Validation artifacts: [training run]({trained_model_result['run_url']}); "
            f"[evaluation run]({evaluation_result['run_url']}). "
            f"Test accuracy: **{evaluation_result['accuracy']:.2f}%**."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You may have noticed a button labeled "Explode". Don't click that, as it will set off a small bomb underneath your humble author's desk in the W&B HQ!

    Just kidding. It "explodes" the graph in a much gentler way:
    `Artifact`s and `Run`s become separated at the level of a single instance,
    rather than a `type`:
    the nodes are not `dataset` and `load-data`, but `dataset:mnist-raw:v1` and `load-data:sunny-smoke-1`, and so on.

    This provides total insight into your pipeline,
    with logged metrics, metadata, and more
    all at your fingertips --
    you're only limited by what you choose to log with us.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verify and next steps

    In the W&B project, open the runs created by this notebook and inspect:

    - the **Artifacts** tab for `mnist-raw`, `mnist-preprocess`, `convnet`, and
      `trained-model`;
    - the Artifact graph to confirm each run consumes the previous step's
      output and produces the next Artifact;
    - the training run charts for `train/loss`, `validation/loss`, and
      `validation/accuracy`;
    - the report run summary for `loss` and `accuracy`, plus the
      `high-loss-examples` media panel.

    Re-run the model initialization or training steps with different
    hyperparameters to create new Artifact versions and compare them in W&B.
    """)
    return


if __name__ == "__main__":
    app.run()
