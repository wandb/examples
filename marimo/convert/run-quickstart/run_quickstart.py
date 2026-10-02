# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "wandb",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App()

with app.setup(hide_code=True):
    import marimo as mo
    import wandb
    import random


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # W&B Quickstart

    Use W&B to track, visualize, and manage machine learning experiments of any size.

    ## Create a machine learning training experiment

    The following example simulates a simple training experiment and logs metrics to W&B.

    First, define the W&B project name and a `config` dictionary. The config stores the input values for the experiment, such as the number of epochs and the learning rate. In this notebook, the form below collects the information for the `config`.

    Next, initialize a W&B run with [`wandb.init()`](https://docs.wandb.ai/models/ref/python/functions/init). The run records the config, metrics, and other information from the training script.

    Inside the training loop, the script simulates an accuracy and loss value for each epoch. It then logs those values to W&B with `run.log()`. After the script runs, you can view the logged metrics in the W&B App.

    ### Authentication

    To save your experiment in W&B, you need to authenticate.

    Authenticate with W&B one of two ways: run **`wandb login`** in
    your shell before starting marimo, or paste your key into the
    **W&B API key** field in the form below. Get your key from
    [wandb.ai/authorize](https://wandb.ai/authorize).
    """)
    return


@app.cell(hide_code=True)
def _():
    epochs = mo.ui.slider(
        start=1,
        stop=20,
        step=1,
        value=10,
        label="Epochs",
        show_value=True,
    )

    lr = mo.ui.slider(
        start=0.001,
        stop=0.1,
        step=0.001,
        value=0.01,
        label="Learning rate",
        show_value=True,
    )
    api_key = mo.ui.text(
        value="",
        kind="password",
        label="W&B API key (blank uses your shell login)",
    )
    project = mo.ui.text(value="my-awesome-project", label="W&B project")
    entity = mo.ui.text(
        value="",
        label="W&B entity \u2014 a team you belong to (blank uses your default)",
    )

    # Batch every control into one form so training only kicks off on submit.
    # `form.value` is None until the user clicks Train model, then becomes a dict
    # keyed by the names below - the training cell gates on that.
    form = (
        mo.md(
            """
            **Training**

            {epochs}

            {lr}

            **W&B run.**

            {api_key}

            {project}

            {entity}
            """
        )
        .batch(
            epochs=epochs,
            lr=lr,
            api_key=api_key,
            project=project,
            entity=entity,
        )
        .form(submit_button_label="Start run", bordered=False)
    )

    form
    return (form,)


@app.cell
def _(form):
    run_path = None

    mo.stop(
        form.value is None,
        mo.md(
            "Configure the run and click Start run to simulate training and log "
            "accuracy and loss to W&B."
        ),
    )

    # Get the data from the form
    cfg = form.value

    # Dictionary with hyperparameters
    config = {
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
    }

    # Authenticate and start the run. Finish any prior run first (marimo keeps the
    # kernel alive across re-submits). A key pasted into the form wins; otherwise
    # fall back to ambient login (shell `wandb login`, WANDB_API_KEY, or netrc).
    # The key is never written to the run config.
    if wandb.run is not None:
        wandb.finish()
    if cfg["api_key"]:
        wandb.login(key=cfg["api_key"])

    with wandb.init(project=cfg["project"], entity=cfg["entity"] or None, config=config) as run:
        offset = random.random() / 5
        print(f"lr: {config['lr']}")

        # Simulate a training run
        for epoch in range(1, config['epochs'] + 1):
            acc = 1 - 2**-config['epochs'] - random.random() / config['epochs'] - offset
            loss = 2**-config['epochs'] + random.random() / config['epochs'] + offset
            print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
            run.log(
                {
                    "epoch": epoch,
                    "accuracy": acc,
                    "loss": loss
                }
            )

        run_path = f"{run.entity}/{run.project}/{run.id}"
    return (run_path,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Get the logged run results from W&B
    """)
    return


@app.cell
def _(run_path):
    history = []
    remote_run = None

    mo.stop(
        run_path is None,
        mo.md("Run an experiment above to load its logged results from W&B."),
    )

    _api = wandb.Api()
    remote_run = _api.run(run_path)

    history = [
        {
            "epoch": row["epoch"],
            "accuracy": round(row["accuracy"], 4),
            "loss": round(row["loss"], 4),
        }
        for row in remote_run.scan_history(
            keys=["epoch", "accuracy", "loss"]
        )
    ]
    return history, remote_run


@app.cell
def _(history, remote_run):
    mo.stop(
        not history,
        mo.md("The run finished, but no metric history is available yet."),
    )

    final = history[-1]

    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"**Run complete:** "
                    f"[{remote_run.name}]({remote_run.url})\n\n"
                    f"Final accuracy: **{final['accuracy']:.2%}**  \n"
                    f"Final loss: **{final['loss']:.4f}**"
                ),
                kind="success",
            ),
            mo.ui.table(history, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
