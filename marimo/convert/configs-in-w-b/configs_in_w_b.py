# /// script
# dependencies = [
#     "wandb==0.30.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(auto_download=["html"])

with app.setup:
    import argparse
    import marimo as mo
    import wandb


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Configs in W&B

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/configs-in-w-b/configs_in_w_b.py/server)
    """)
    return


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

    ## Quickstart
    Use [Weights & Biases](https://wandb.ai)
    for machine learning experiment tracking, dataset versioning, and project collaboration.

    <div><img /></div>

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

    <div><img /></div>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authentication

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

    Set [`wandb.config`](https://docs.wandb.ai/guides/track/config)
    once at the beginning of your script to save your training configuration: hyperparameters, input settings like dataset name or model type, and include any other independent variables or metadata for your experiments.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Why does that matter?

    This is useful for analyzing your experiments and reproducing your work in the future. You'll be able to group by `config` values in our web interface, comparing the settings of different runs and seeing how these affect the output.

    > Note that output metrics or dependent variables (like loss and accuracy) should be saved with `wandb.log` instead.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How do I set up a `config`?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Your `config` should be set just once at the beginning of your training experiment.

    But workflows differ, so we offer a number of ways to set up your config.

    Let's look at all the ways you can create and send the config dictionary to the Dashboard!
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Setting the `config` at `init`ialization

    The best time to set the `config` values is when you call [`wandb.init`](https://docs.wandb.ai/guides/track/launch),
    by passing a dictionary as the `config` keyword argument.
    """)
    return


@app.cell(hide_code=True)
def _():
    create_config_run = mo.ui.run_button(
        label="Create the W&B config example run"
    )
    mo.vstack(
        [
            mo.md(
                "This creates one run in the `config_example` project. Use the "
                "controls below to apply each config update separately."
            ),
            create_config_run,
        ]
    )
    return (create_config_run,)


@app.cell
def _(create_config_run, wandb_settings):
    mo.stop(
        not create_config_run.value,
        mo.callout(
            mo.md("Click the button above when you're ready to create the example run."),
            kind="info",
        ),
    )

    if wandb.run is not None:
        wandb.finish()

    config_run = wandb.init(
        project=wandb_settings["project"],
        entity=wandb_settings["entity"],
        config={"dataset": "CelebA", "type": "baseline"},
    )
    config_run_url = config_run.url
    mo.callout(
        mo.md(f"Run created: [open it in W&B]({config_run_url})."),
        kind="success",
    )
    return (config_run,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Open the Run page from the link shown above and head to the [Overview tab](https://docs.wandb.ai/ref/app/pages/run-page#overview-tab)
    (top of the list of panels on the left-most side of the screen).
    You'll see a "Config" section that looks like this:
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
    You give us a (possibly nested) dictionary as your `config`, and we'll flatten the names using dots in our backend.

    > _Side Note_: We recommend that you avoid using dots in your config variable names, and use a dash or underscore instead. Once you've created your `config` dictionary, if your script accesses `wandb.config` keys below the root, use the dictionary access syntax, `["key"]["foo"]`, instead of the attribute access syntax, `config.key.foo`.
    """)
    return


@app.cell(hide_code=True)
def _():
    add_config_parameters = mo.ui.run_button(
        label="Add parameters to the W&B config"
    )
    mo.vstack(
        [
            mo.md(r"""
    ### Adding to the `config` by hand
    You can add more parameters to the `config` later if you want:
    """),
            add_config_parameters,
        ]
    )
    return (add_config_parameters,)


@app.cell
def _(add_config_parameters, config_run):
    mo.stop(
        not add_config_parameters.value,
        mo.callout(
            mo.md(
                "Click the button above to add `epochs` and `batch_size` to "
                "this run's config."
            ),
            kind="info",
        ),
    )

    config_run.config.epochs = 4
    config_run.config["batch_size"] = 32

    manual_config_update = True
    dict(config_run.config)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now, your Config section on the dashboard has been updated:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <img src="https://i.imgur.com/cnvEuSR.png" width="450"/>
    """)
    return


@app.cell(hide_code=True)
def _():
    add_argparse_parameters = mo.ui.run_button(
        label="Add argparse parameters to the W&B config"
    )
    mo.vstack(
        [
            mo.md(r"""
    ### Adding to the `config` with `argparse`

    `config` is a dictionary-like object, and it can be built from lots of dictionary-like objects.

    For example, you can pass in the arguments object produced by `argparse`.
    [`argparse`](https://docs.python.org/3/library/argparse.html), short for `arg`ument `parse`r, is a standard library module in Python 3.2 and above that makes it easy to write scripts that take advantage of all the flexibility and power of command line arguments. And it's Pythonic!

    This is especially convenient for tracking results from scripts that are launched from the command line.
    """),
            add_argparse_parameters,
        ]
    )
    return (add_argparse_parameters,)


@app.cell
def _(add_argparse_parameters, config_run):
    mo.stop(
        not add_argparse_parameters.value,
        mo.callout(
            mo.md("Click the button above to add the parsed arguments to this run's config."),
            kind="info",
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_per_gpu', type=int, default=8,
                        help='input batch size for training (default: 8)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.1,
                        help='weight decay (default: 0.1)')

    args = parser.parse_args(args=[])
    config_run.config.update(args)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here's the updated Config panel:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <img src="https://i.imgur.com/zWSpGNy.png" width=450>
    """)
    return


@app.cell(hide_code=True)
def _():
    update_config_with_api = mo.ui.run_button(
        label="Update the W&B config with the Public API"
    )
    mo.vstack(
        [
            mo.md(r"""
    ### Updating the `config` with the API

    What if your run has finished, but you realized you forgot to log something?

    Never fear, you can always use the
    [public API](https://docs.wandb.ai/ref/python/public-api)
    to update your `config`
    (or anything else about your run!)
    at any time. You just need to know the details of the `run` you want to update.
    """),
            update_config_with_api,
        ]
    )
    return (update_config_with_api,)


@app.cell
def _(config_run, update_config_with_api):
    mo.stop(
        not update_config_with_api.value,
        mo.callout(
            mo.md("Click the button above to add `bar` through the W&B Public API."),
            kind="info",
        ),
    )

    api = wandb.Api()

    # pulling the relevant info automatically from the run object
    # this can also be found on the website
    username = config_run.entity
    project = config_run.project
    run_id = config_run.id

    api_run = api.run(f"{username}/{project}/{run_id}")
    api_run.config["bar"] = 32
    api_run.update()
    return


@app.cell
def _():
    mo.md(r"""
    Here's what the final Config panel looks like:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <img src="https://i.imgur.com/mxmIbyK.png" width=450>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Using `config` for great good!
    The `config` parameters are useful for performing grouping, filtering, and aggregating on your experiments and their results.

    ### The examples below come from the project [here](https://wandb.ai/wandb/DistHyperOpt).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Filtering Runs
    Filter tab allows you to display the runs that quality one or more conditions. These conditions can be formed by applying relational operators to any of the parameters logged in the `config` file.

    [Our example project](https://wandb.ai/wandb/DistHyperOpt) compares various hyper-parameter tuning methods and has more than 80 runs. Each run has a "Job Type" logged in the `config` which corresponds

    Let's say you want to visualize only the ones that are generated by a particular tuning algorithm, like Population Based Traing (`pbt`). You can do that by applying a filter on "Job Type".

    Run the cell below to see this in action!
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.Html(r"""
    <iframe
      width="100%"
      height="360"
      src="https://www.youtube.com/embed/aSMXwOSPtJE?rel=0"
      title="Filter W&B runs by config values"
      frameborder="0"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
      referrerpolicy="strict-origin-when-cross-origin"
      allowfullscreen>
    </iframe>
    <p><a href="https://www.youtube.com/watch?v=aSMXwOSPtJE" target="_blank" rel="noopener noreferrer">Open the video on YouTube</a></p>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Grouping Runs
    You can group your experiments in the dashboard of your project based on a particular column from `config`. A common use case for this would be grouping sub-experiments within a larger project.

    Our runs are grouped based on "Job Type". The Group tab is located next to the Filter Tab. You can group your runs by any parameter present in the config.

    ![Imgur](https://i.imgur.com/gTLKRP7.png?1)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Parallel Coordinates Chart

    Often, the main thing we want to do with a group of Runs is make comparisons.

    The W&B Dashboard includes a Chart type for exactly this purpose:
    the Parallel Coordinates chart.

    A Parallel Coordinates chart represents each Run in the group as a line.
    This line passes through as many of the `config` values
    or logged metrics as you like,
    and is colored by its value on a single metric.
    This lets you take in, at a glance,
    which hyperparameter configurations were most and least successful.
    See the example below.

    Head to a [group of Runs in this project](https://wandb.ai/wandb/DistHyperOpt/groups/dcgan_train)
    and build a Parallel Coordinates chart like the one pictured below
    by
    1. clicking the + sign in the top-right corner, aligned with "Charts",
    2. selecting "Parallel Coordinates" from the available Charts, and
    3. adding the columns in the image, in order.

    ![Imgur](https://i.imgur.com/ugrpq9K.png)
    """)
    return


if __name__ == "__main__":
    app.run()
