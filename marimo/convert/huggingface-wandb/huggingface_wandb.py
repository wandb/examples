# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "accelerate",
#     "datasets",
#     "evaluate",
#     "marimo>=0.24.2",
#     "scikit-learn==1.9.1",
#     "scipy==1.18.1",
#     "torch==2.14.1",
#     "transformers==4.57.6",
#     "wandb==0.30.0",
# ]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(auto_download=["html"])

with app.setup:
    import marimo as mo

    import os
    import subprocess
    import sys
    import tempfile
    from importlib.metadata import version
    from pathlib import Path
    from urllib.request import urlopen

    # The training subprocess needs these packages in the notebook's interpreter.
    # Explicit imports also expose them to marimo's package management.
    import accelerate
    import datasets
    import evaluate
    import torch
    import transformers
    import wandb


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <style>
    .wandb-config-logo--dark {
      display: none;
    }

    body.dark .wandb-config-logo--light {
      display: none;
    }

    body.dark .wandb-config-logo--dark {
      display: block;
    }
    </style>

    <img class="wandb-config-logo--light" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_blackwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />
    <img class="wandb-config-logo--dark" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_goldwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />

    # Hugging Face + Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/huggingface-wandb/huggingface_wandb.py/server)

    This notebooks shows how to train and monitor Hugging Face models with Weights & Biases. Microsoft Research Paraphrase Corpus (MRPC) provides labeled sentence pairs and is part of the GLUE language-understanding benchmark.

    We will use Hugging Face's `run_glue.py` script to handle data preparation and training. Its built-in `Trainer` integration sends metrics to Weights & Biases when you pass `--report_to wandb`. The Trainer callback handles metric logging for you.

    A GPU is recommended for the full BERT run below. The first run downloads
    the pretrained model and MRPC data. CPU execution works but takes longer.

    <img src="https://i.imgur.com/vnejHGh.png" width="800" alt="Example of Hugging Face training metrics in Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run with the notebook's dependencies

    From this notebook's directory, launch it with its declared Python and package versions:

    ```bash
    uvx marimo edit --sandbox huggingface_wandb.py
    ```

    The training script uses the same Python interpreter as the notebook.
    Opening this file in an existing marimo environment does not install the
    dependencies listed at the top of the file. That environment must also
    contain `evaluate`, `accelerate`, `datasets`, PyTorch, and Transformers.
    The sandbox command installs the declared dependencies in an isolated environment.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `huggingface-demo`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="huggingface-demo", label="Project", full_width=True),
    ).form(submit_button_label="Connect to Weights & Biases", bordered=True)
    wandb_setup_form
    return (wandb_setup_form,)


@app.cell(hide_code=True)
def _(wandb_setup_form):
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

    entity = _entity
    project = _project
    mo.md("✓ Connected to Weights & Biases.")
    return entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Download the training script

    Download `run_glue.py` from the same Transformers release installed by this
    notebook. Matching the versions keeps the script's arguments and APIs
    compatible with the library. The script is saved in temporary storage so
    it can be launched in the next steps.
    """)
    return


@app.cell
def _():
    transformers_version = version("transformers")
    run_glue_url = (
        "https://raw.githubusercontent.com/huggingface/transformers/"
        f"v{transformers_version}/examples/pytorch/text-classification/run_glue.py"
    )
    run_glue_path = Path(tempfile.mkdtemp(prefix="huggingface-glue-")) / "run_glue.py"

    with urlopen(run_glue_url, timeout=60) as source:
        run_glue_path.write_bytes(source.read())
    return (run_glue_path,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configure Weights & Biases for the training process

    The script runs in a separate Python process. The next cell copies the runtime
    environment and adds your selected project and team. Existing environment
    credentials are inherited, and credentials saved by the SDK are available
    to the child process in the same runtime.

    `WANDB_WATCH="all"` adds parameter and gradient histograms to the run.
    Set it to `"false"` if you only need metrics and want to reduce logging
    overhead. The Trainer creates the Weights & Biases run when training starts.
    """)
    return


@app.cell
def _(entity, project):
    run_environment = os.environ.copy()
    run_environment["WANDB_PROJECT"] = project
    run_environment["WANDB_WATCH"] = "all"
    if entity is not None:
        run_environment["WANDB_ENTITY"] = entity
    return (run_environment,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fine-tune BERT on MRPC

    The arguments below select `bert-base-uncased`, tokenize sentence pairs to
    at most 256 tokens, and train for three epochs with a batch size of 32 and
    learning rate of `2e-4`. `--logging_steps 50` records training metrics every
    50 optimizer steps, while `--eval_strategy epoch` evaluates on the validation
    set after each epoch. MRPC evaluation reports accuracy and F1.

    `--report_to wandb` enables experiment tracking. The process prints a link
    to the run as training begins. It uses the same Python environment as the
    notebook and automatically selects an available accelerator or CPU.

    Each execution saves the final model and metric files in a new temporary
    output directory. For a quick check, add `--max_train_samples 64` and
    `--max_eval_samples 32`, reduce `--num_train_epochs` to `1`, and set
    `--logging_steps` to `1`. Those small samples help check that the workflow runs. Use a larger, fixed
    validation set when comparing model quality.
    """)
    return


@app.cell
def _(run_environment, run_glue_path):
    output_dir = tempfile.mkdtemp(prefix="bert-mrpc-")
    training_command = [
        sys.executable,
        str(run_glue_path),
        "--model_name_or_path", "bert-base-uncased",
        "--task_name", "mrpc",
        "--do_train",
        "--do_eval",
        "--max_seq_length", "256",
        "--per_device_train_batch_size", "32",
        "--learning_rate", "2e-4",
        "--num_train_epochs", "3",
        "--logging_steps", "50",
        "--eval_strategy", "epoch",
        "--save_strategy", "no",
        "--output_dir", output_dir,
        "--run_name", "bert-base-mrpc",
        "--report_to", "wandb",
    ]

    subprocess.run(training_command, env=run_environment, check=True)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify the run and compare experiments

    Open the run link printed by Weights & Biases during training, or find the run named
    `bert-base-mrpc` in your chosen project. In the run's workspace, inspect
    `train/loss`, `eval/loss`, `eval/accuracy`, and `eval/f1`. The training step
    is the horizontal axis, and evaluation points appear after each epoch.
    Check the run's config for settings such as `learning_rate`,
    `num_train_epochs`, and `per_device_train_batch_size`. The system metrics
    show resource usage. Parameter and gradient histograms are logged
    periodically when `WANDB_WATCH` is `"all"`.

    Try another learning rate or change `--model_name_or_path` to
    `distilbert-base-uncased`, give the experiment a descriptive `--run_name`,
    and rerun training. Compare the runs' validation accuracy and F1 alongside
    training time in the project workspace.

    See the [Hugging Face + Weights & Biases Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU)
    for an example dashboard, and the
    [BERT versus DistilBERT report](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU)
    for a comparison of architectures.
    """)
    return


if __name__ == "__main__":
    app.run()
