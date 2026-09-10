# /// script
# dependencies = [
#     "accelerate",
#     "datasets",
#     "evaluate",
#     "fsspec==2026.6.0",
#     "transformers @ git+https://github.com/huggingface/transformers",
#     "wandb==0.29.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(auto_download=["html"])

with app.setup(hide_code=True):
    import marimo as mo

    import os
    import uuid
    import subprocess
    import wandb
    import fsspec

    # Optional: log both gradients and parameters
    os.environ['WANDB_WATCH'] = 'all'


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Hugging Face + W&B

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/huggingface-wandb/huggingface_wandb.py/server)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Visualize your [Hugging Face](https://github.com/huggingface/transformers) model's performance quickly with a seamless [W&B](https://wandb.ai/site) integration.

    Compare hyperparameters, output metrics, and system stats like GPU utilization across your models.

    <img src="https://i.imgur.com/vnejHGh.png" width="800" alt="Hugging Face and Weights & Biases integration" />
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md(r"""
    <style>
    .wandb-by-cw-logo--dark {
      display: none;
    }

    :host-context(body.dark) .wandb-by-cw-logo--light {
      display: none;
    }

    :host-context(body.dark) .wandb-by-cw-logo--dark {
      display: block;
    }
    </style>

    <img class="wandb-by-cw-logo--light" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_blackwhite.svg" width="320" alt="Weights & Biases by CoreWeave" />
    <img class="wandb-by-cw-logo--dark" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_goldwhite.svg" width="320" alt="Weights & Biases by CoreWeave" />

    Use [Weights & Biases](https://wandb.com) for machine learning experiment tracking, dataset versioning, and project collaboration.

    <img src="https://wandb.me/mini-diagram" width="600" alt="Weights & Biases features" />

    - **Unified dashboard**: Central repository for all your model metrics and predictions
    - **Lightweight**: No code changes required to integrate with Hugging Face
    - **Accessible**: Free for individuals and academic teams
    - **Secure**: All projects are private by default
    - **Trusted**: Used by machine learning teams at OpenAI, Toyota, Lyft and more

    Think of W&B like GitHub for machine learning models— save machine learning experiments to your private, hosted dashboard. Experiment quickly with the confidence that all the versions of your models are saved for you, no matter where you're running your scripts.

    W&B lightweight integrations works with any Python script, and all you need to do is sign up for a free W&B account to start tracking and visualizing your models.
    """),
        kind="neutral",
        title="🤔 Why should I use W&B?",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 🚀 Getting started

    In this tutorial, we work with the Hugging Face and Weights & Biases libraries, and the GLUE dataset and training script.
    - [Hugging Face Transformers](https://github.com/huggingface/transformers): Natural language models and datasets
    - [Weights & Biases](https://docs.wandb.com/): Experiment tracking and visualization
    - [GLUE dataset](https://gluebenchmark.com/): A language understanding benchmark dataset
    - [GLUE script](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py): Model training script for sequence classification
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
    In the Hugging Face Transformers repo, we've instrumented the Trainer to automatically log training and evaluation metrics to W&B at each logging step.

    Here's an in depth look at how the integration works: [Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 🔑 Authentication

    Enter your [W&B API key](https://wandb.ai/authorize) and, if needed, your team or entity. You can leave the key blank when this environment already has W&B credentials.
    """)
    return


@app.cell(hide_code=True)
def _():
    _api_key_input = mo.ui.text(
        kind="password",
        label="W&B API key (optional)",
        placeholder="Paste a key or use cached credentials",
        full_width=True,
    )
    _entity_input = mo.ui.text(
        label="W&B entity or team (optional)",
        placeholder="Leave blank to use your default entity",
        full_width=True,
    )
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}")
        .batch(api_key=_api_key_input, entity=_entity_input)
        .form(submit_button_label="Connect to W&B", bordered=True)
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
    Optionally, we can set environment variables to customize W&B logging. See [documentation](https://docs.wandb.com/library/integrations/huggingface).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 👟 Train the model
    Next, call the downloaded training script [run_glue.py](https://huggingface.co/transformers/examples.html#glue) and see training automatically get tracked to the Weights & Biases dashboard. This script fine-tunes BERT on the Microsoft Research Paraphrase Corpus— pairs of sentences with human annotations indicating whether they are semantically equivalent.
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


@app.cell
def _(run_environment, run_glue_path, task_name):
    subprocess.run(
        [
            "python",
            run_glue_path,
            "--model_name_or_path",
            "bert-base-uncased",
            "--task_name",
            task_name,
            "--do_train",
            "--do_eval",
            "--max_seq_length",
            "256",
            "--per_device_train_batch_size",
            "32",
            "--learning_rate",
            "2e-4",
            "--num_train_epochs",
            "3",
            "--output_dir",
            f"/tmp/{task_name}/",
            "--report_to",
            "wandb",
            "--logging_steps",
            "50",
        ],
        env=run_environment,
        check=True,
    )
    return


@app.cell(hide_code=True)
def _(wandb_run_url):
    mo.md(f"""
    ## 👀 Visualize results in dashboard

    [**Open this training run in W&B ↗**]({wandb_run_url})

    Click the link above, or go to [wandb.ai](https://app.wandb.ai) to see your results stream in live. The link to see your run in the browser will appear after all the dependencies are loaded — look for the following output: "**wandb**: 🚀 View run at [URL to your unique run]"

    **Visualize Model Performance**
    It's easy to look across dozens of experiments, zoom in on interesting findings, and visualize highly dimensional data.

    ![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M79Y5aLAFsMEcybMZcC%2F-M79YL90K1jiq-3jeQK-%2Fhf%20gif%2015.gif?alt=media&token=523d73f4-3f6c-499c-b7e8-ef5be0c10c2a)

    **Compare Architectures**
    Here's an example comparing [BERT vs DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU) — it's easy to see how different architectures effect the evaluation accuracy throughout training with automatic line plot visualizations.
    ![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M79Y5aLAFsMEcybMZcC%2F-M79Ytpj6q6Jlv9RKZGT%2Fgif%20for%20comparing%20bert.gif?alt=media&token=e3dee5de-d120-4330-b4bd-2e2ddbb8315e)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 📈 Track key information effortlessly by default
    Weights & Biases saves a new run for each experiment. Here's the information that gets saved by default:
    - **Hyperparameters**: Settings for your model are saved in Config
    - **Model Metrics**: Time series data of metrics streaming in are saved in Log
    - **Terminal Logs**: Command line outputs are saved and available in a tab
    - **System Metrics**: GPU and CPU utilization, memory, temperature etc.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 🤓 Learn more!
    - [Documentation](https://docs.wandb.ai/tutorials/huggingface/): docs on the Weights & Biases and Hugging Face integration
    - [Videos](http://wandb.me/youtube): tutorials, interviews with practitioners, and more on our YouTube channel
    - Contact: Message us at contact@wandb.com with questions
    """)
    return


if __name__ == "__main__":
    app.run()
