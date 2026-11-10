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

    Run the following cell. When prompted, specify your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys):
    """)
    return


@app.cell(hide_code=True)
def _():
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
def _():
    mo.md(r"""
    After submitting the API key, run the following cell to verify the
    credentials and load your available Weights & Biases entities.
    """)
    return


@app.cell(hide_code=True)
def _(wandb_api_form):
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
        mo.md(
            "✓ Credentials verified. Choose a personal or team entity below."
        ),
        kind="success",
    )
    return (available_entities,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set your team and project

    Run the following cell and specify your:

    - Weights & Biases personal or team entity.
    - A name for your [project](https://docs.coreweave.com/products/wandb/track/project-page#projects). The default project is `huggingface-demo`.

    Then select **Use this Weights & Biases target**. A project does not need to exist yet;
    Weights & Biases creates it when the first run starts if you have write access.
    """)
    return


@app.cell(hide_code=True)
def _(available_entities):
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
                value="huggingface-demo",
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
def _():
    mo.md(r"""
    After submitting the target form, run the next cell to confirm the selected
    destination. Weights & Biases verifies project write access when training starts.
    """)
    return


@app.cell(hide_code=True)
def _(wandb_setup_form):
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _entity = _submission["entity"]
    _project = _submission["project"].strip()

    entity = _entity
    project = _project

    mo.callout(
        mo.md(
            f"✓ Using `{entity}/{project}`. Weights & Biases will confirm write access "
            "when training starts."
        ),
        kind="success",
    )
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

    From the previous cell's output, select the link to your project or run that is returned by Weights & Biases during training. Alternatively, navgiate to Weights & Biases App and search for **huggingface-demo** (or the name of the project if you changed it).


    It should look similar to the following:

    ```
    wandb: ⭐️ View project at https://wandb.ai/team/huggingface-demo
    wandb: 🚀 View run at https://wandb.ai/team/huggingface-demo/runs/uz6s7b1k
    ```

    Where **team** is the name you specified for **Weights & Biases entity** in the form above and `uz6s7b1k` is the unique [run ID](https://docs.coreweave.com/products/wandb/runs/run-identifiers#run-id).

    Navigate to your project's workspace in the Weights & Biases App. Scroll to the `train` panel. You should see the following plots with the following titles:

    - `train/loss`
    - `train/learning_rate`

    Similarly, within the **eval** section, you shold see:

    - `eval/loss`
    - `eval/accuracy`

    View the [run's config](https://docs.coreweave.com/products/wandb/track/config#view-config-values) to view the parameters that where used as inputes such as `learning_rate`,
    `num_train_epochs`, and `per_device_train_batch_size`.

    USe the [run's summary metrics](https://docs.coreweave.com/products/wandb/track/log/log-summary#view-summary-metrics) to view the model's outputs.

    ## Next steps

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
