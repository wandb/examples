# /// script
# dependencies = ["marimo"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(app_title="Deprecated: OpenAI API Autologger Quickstart")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OpenAI API Autologger Quickstart — Deprecated

    **This tutorial is retired. Use
    [W&B Weave's OpenAI integration](https://docs.wandb.ai/weave/guides/integrations/openai)
    for automatic logging in new applications.**

    The original Weights & Biases lesson used
    `wandb.integration.openai.autolog` to record two chat conversations, model
    responses, token usage, and latency in a W&B run. It then disabled logging
    with `autolog.disable()`.

    That integration depends on the legacy `openai.ChatCompletion.create()`
    interface. [W&B's compatibility documentation](https://docs.wandb.ai/models/integrations/openai-api)
    limits it to OpenAI Python SDK 0.28.1 and earlier. This notebook no longer
    installs those libraries or executes the old API calls.

    ## Continue with Weave

    Weave automatically traces supported OpenAI calls after `weave.init()`.
    Follow the [OpenAI integration guide](https://docs.wandb.ai/weave/guides/integrations/openai)
    to capture inputs, outputs, latency, and token usage with the modern SDK,
    then inspect those calls in Weave.

    See the [official OpenAI SDK documentation](https://platform.openai.com/docs/libraries)
    for current client setup. Chat Completions remains available; this retirement
    concerns the old Python SDK interface and W&B autologger.

    ## Historical reference

    The [original W&B tutorial source](https://github.com/wandb/examples/blob/main/colabs/openai/OpenAI_API_Autologger_Quickstart.ipynb)
    is retained for historical context. Its API calls and dependency assumptions
    should not be copied into a new project.
    """)
    return


if __name__ == "__main__":
    app.run()
