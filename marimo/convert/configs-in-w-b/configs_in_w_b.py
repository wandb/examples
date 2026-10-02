# /// script
# dependencies = [
#     "wandb==0.30.0",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(auto_download=["html"])

with app.setup:
    import argparse
    import marimo as mo
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

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/configs-in-w-b/configs_in_w_b.py/server)

    ## Quickstart
    Use [Weights & Biases](https://wandb.ai)
    for machine learning experiment tracking, dataset versioning, visualizations, and project collaboration.


    This notebook demonstrates how to use a W&B Run's `config` property to save your training configuration:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authentication

    Run the following cell and privde your [Forge API key](https://wandb.ai/authorize) and your team name. Then select **Connect to W&B**.
    """)
    return


@app.cell(hide_code=True)
def _():
    _api_key_input = mo.ui.text(
        kind="password",
        label="Forge API key (optional)",
        placeholder="Paste a key or use cached credentials",
        full_width=True,
    )
    _entity_input = mo.ui.text(
        label="Forge team entity",
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
            mo.md("Connect to W&B above before creating the example run."),
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
        "project": "config_example",
        "entity": _entity or None,
    }
    mo.callout(mo.md("Connected to W&B."), kind="success")
    return (wandb_settings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What's a `config` for?

    Use [`wandb.Run.config`](https://docs.wandb.ai/guides/track/config) property
    once at the beginning of your script to save your training configuration:
    - hyperparameters
    - input settings like dataset name or model type
    - other independent variables or metadata for your experiments

    This is useful for analyzing your experiments and reproducing your work in the future. You can [group](https://docs.coreweave.com/products/wandb/runs/grouping), [filter](https://docs.coreweave.com/products/wandb/runs/filter-runs), by `config` values using the W&B App or programmatically.

    > Note that output metrics or dependent variables (like loss and accuracy) should be saved with [`wandb.Run.log()`](https://docs.coreweave.com/products/wandb/ref/python/experiments/run) instead.

    ## How do I set up a `config`?

    Pass a dictionary of key-value pairs to the `config` paramater when you call [`wandb.init()`](https://docs.coreweave.com/products/wandb/ref/python/functions/init).

    > Configurations are typically defined in the beginning of a training script. Machine learning workflows may vary, however, so you are not required to define a configuration at the beginning of your training script.


    ## Example: Define config when you initialize a run

    The following cell passes an. configuration at the beginning of their experiment. It contains two to key-value pairs: `dataset:CelebA` and `type:baseline`.
    """)
    return


@app.cell
def _(wandb_settings):
    config = {
        "dataset": "CelebA", 
        "type": "baseline",
    }


    with wandb.init(
        project=wandb_settings["project"],
        entity=wandb_settings["entity"],
        config=config,
    ) as config_run:
        config_run_url = config_run.url
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The previous cell returns a link to [run's](https://docs.coreweave.com/products/wandb/runs) **Overview** page in the W&B App UI.

    Select the link and navigate to the **Config** section. If you ran the previous cell as is, it should look similar to the following:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <img src="https://i.imgur.com/nAC9KEF.png" width="450"/>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Use dashes (-) or underscores (_) instead of periods (.) in your config variable names. For more information, see [Configure experiments](https://docs.coreweave.com/products/wandb/track/config).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example: Update an existing config

    Use the [W&B Public API](https://docs.coreweave.com/products/wandb/ref/python/public-api) to update a completed run’s config.
    You must provide the API with your team entity, the name of the project the run was logged to, and the [run’s ID](https://docs.coreweave.com/products/wandb/runs/run-identifiers#run-id).


    ```
    wandb: setting up run
    ```

    This is the run's ID. Copy it and replace "run_id" in the following cell.

    > You can also find a [run's ID programmatically or with the W&B App](https://docs.coreweave.com/products/wandb/runs/run-identifiers#find-a-run%E2%80%99s-id).

    The following cell adds a new config value (`bar`) to your existing run object.
    """)
    return


@app.cell
def _(wandb_settings):
    run_id = "iomdo1ui" # replace with the run ID you want to update

    api = wandb.Api()
    api_run = api.run(f"{wandb_settings["entity"]}/{wandb_settings["project"]}/{run_id}")


    api_run.config["epochs"] = 4
    api_run.config['batch_size'] = 32
    api_run.update()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps:

    There are more ways to create, update, and mange config values for your experiments. Read the [Configure experiments documentation](https://docs.coreweave.com/products/wandb/track/config) to learn more.


    For a live demo, explore this example [CoreWeave Project](https://wandb.ai/wandb/DistHyperOpt) to experiment with.
    """)
    return


if __name__ == "__main__":
    app.run()
