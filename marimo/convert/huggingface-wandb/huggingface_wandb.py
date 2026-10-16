# /// script
# dependencies = [
#     "accelerate",
#     "datasets",
#     "evaluate",
#     "fsspec[http]==2026.6.0",
#     "marimo",
#     "scikit-learn==1.9.1",
#     "scipy==1.18.1",
#     "torch==2.14.1",
#     "transformers @ git+https://github.com/huggingface/transformers",
#     "wandb==0.29.0",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(auto_download=["html"])

with app.setup:
    import marimo as mo

    import os
    import sys
    import uuid
    import subprocess
    import torch
    import wandb
    import fsspec
    import sklearn
    import scipy

    # Optional: log both gradients and parameters
    os.environ['WANDB_WATCH'] = 'all'


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Hugging Face + Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/huggingface-wandb/huggingface_wandb.py/server)

    <!-- Compare hyperparameters, output metrics, and system stats like GPU utilization across your models. -->


    Fine-tune a BERT model for sentence-pair classification and use the W&B integration for Hugging Face Transformers to track the experiment.

    This tutorial uses the Microsoft Research Paraphrase Corpus (MRPC), which contains pairs of sentences labeled according to whether they have the same meaning. MRPC is one of the tasks in the General Language Understanding Evaluation (GLUE) benchmark.

    The tutorial uses:

    - [Hugging Face Transformers](https://github.com/huggingface/transformers): Natural language models and datasets
    - [Weights & Biases](https://docs.wandb.com/): Experiment tracking and visualization
    - [GLUE](https://gluebenchmark.com/): A benchmark for evaluating language-understanding models
    - [`run_glue.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py): A Hugging Face script for training sequence-classification models

    <img src="https://i.imgur.com/vnejHGh.png" width="800" alt="Hugging Face and Weights & Biases integration" />

    See the accompanying [Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authentication

    Connect the notebook to W&B so the Hugging Face Trainer can record the experiment. Enter your [Forge API key](https://wandb.ai/authorize) and optional team name, then select **Connect to Weights & Biases**. If you omit the team, W&B uses your default entity.
    """)
    return


@app.cell(hide_code=True)
def _():
    _api_key_input = mo.ui.text(
        kind="password",
        label="Forge API key",
        placeholder="Paste a key or use cached credentials",
        full_width=True,
    )
    _entity_input = mo.ui.text(
        label="Forge team name",
        placeholder="Leave blank to use your default entity",
        full_width=True,
    )
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}")
        .batch(api_key=_api_key_input, entity=_entity_input)
        .form(submit_button_label="Connect to Weights & Biases", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            mo.md("Connect to W&B above before running the training example."),
            kind="info",
        ),
    )

    _api_key = wandb_login_form.value["api_key"].strip()
    _entity = wandb_login_form.value["entity"].strip()
    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
        _login_error = None
    except wandb.errors.Error as _error:
        _login_ok = False
        _login_error = str(_error)

    mo.stop(
        not _login_ok,
        mo.callout(
            mo.md(
                "W&B authentication did not complete. "
                f"Check the API key and try again. \n\nW&B reported: `{_login_error or 'unknown error'}`"
            ),
            kind="danger",
        ),
    )

    wandb_settings = {
        "project": "huggingface-demo",
        "entity": _entity or None,
    }
    mo.callout(
        mo.md("Connected to W&B."),
        kind="success",
    )
    return (wandb_settings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Download script

    Download Hugging Face's `run_glue.py` script to the notebook environment. The script loads a GLUE task, fine-tunes a sequence-classification model, and evaluates the trained model.
    """)
    return


@app.cell
def _():
    run_glue_url = (
        "https://raw.githubusercontent.com/huggingface/transformers/"
        "refs/heads/main/examples/pytorch/text-classification/run_glue.py"
    )
    run_glue_path = "run_glue.py"

    with fsspec.open(run_glue_url, "rb") as _source:
        with open(run_glue_path, "wb") as _destination:
            _destination.write(_source.read())
    return (run_glue_path,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train the model

    Fine-tune `bert-base-uncased` to determine whether two sentences have the same meaning. Before starting the training process, the notebook checks for a GPU and prepares the W&B run configuration.
    """)
    return


@app.cell(hide_code=True)
def _():
    gpu_available = torch.cuda.is_available()

    (
        mo.callout(
            mo.md(
                f"GPU ready: **{torch.cuda.get_device_name(0)}**. "
                "The training script will use it automatically."
            ),
            kind="success",
        )
        if gpu_available
        else mo.callout(
            mo.md(
                "No GPU is attached to this session. In Molab, click the "
                "notebook specs button in the header, attach a GPU, then save "
                "and restart. Reconnect to W&B after the restart. Training is "
                "paused to prevent an unexpectedly slow CPU run."
            ),
            kind="warn",
            title="GPU not available",
        )
    )
    return (gpu_available,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Configure the training run

    The notebook runs `run_glue.py` as a separate Python process. Because that process cannot access this notebook's Python variables directly, the following cell passes the W&B project, team, and run ID through environment variables.

    The cell also selects MRPC as the GLUE task and constructs the run URL. It prepares the run configuration but does not create the W&B run. The Hugging Face Trainer creates the run when logging begins.
    """)
    return


@app.cell
def _(wandb_settings):
    task_name = "MRPC"
    wandb_run_id = uuid.uuid4().hex
    wandb_run_entity = (
        wandb_settings["entity"] or wandb.Api().default_entity
    )

    run_environment = os.environ.copy()
    run_environment.update(
        {
            "WANDB_RUN_ID": wandb_run_id,
            "WANDB_ENTITY": wandb_run_entity,
            "WANDB_PROJECT": wandb_settings["project"],
        }
    )
    wandb_run_url = wandb.Settings(
        entity=wandb_run_entity,
        project=wandb_settings["project"],
        run_id=wandb_run_id,
    ).run_url
    return run_environment, task_name, wandb_run_url


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Start training

    Launch the downloaded training script with the model, task, and training options shown below.

    The `--report_to wandb` option enables the Trainer's built-in W&B integration. The command waits for training and evaluation to finish, saves the model output in temporary notebook storage, and reports an error if the script fails.
    """)
    return


@app.cell
def _(gpu_available, run_environment, run_glue_path, task_name):
    mo.stop(not gpu_available)

    training_command = [
        sys.executable,
        run_glue_path,

        # Model and task
        "--model_name_or_path", "bert-base-uncased",
        "--task_name", task_name,
        "--do_train",
        "--do_eval",

        # Training configuration
        "--max_seq_length", "256",
        "--per_device_train_batch_size", "32",
        "--learning_rate", "2e-4",
        "--num_train_epochs", "3",
        "--logging_steps", "50",

        # Output and experiment tracking
        "--output_dir", f"/tmp/{task_name}/",
        "--report_to", "wandb",
    ]

    subprocess.run(
        training_command,
        env=run_environment,
        check=True,
    )
    return


@app.cell(hide_code=True)
def _(wandb_run_url):
    mo.md(f"""
    ## View the results

    Open the completed run to review its training and evaluation metrics, inspect system metrics, and compare it with other experiments:

    [**View the training run in W&B**]({wandb_run_url})

    ## Next step: Compare architectures

    Explore a W&B report that [compares BERT and DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU) to see how model architecture affects evaluation accuracy during training.
    """)
    return


if __name__ == "__main__":
    app.run()
