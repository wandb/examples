# /// script
# dependencies = ["openai", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/openai/OpenAI_API_Autologger_Quickstart.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{openai-autologger-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{openai-autologger-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🏃‍♀️ OpenAI API Logger
    Use the **[Weights & Biases](https://wandb.ai/site?utm_source=openai_autologger_colab&utm_medium=code&utm_campaign=openai_autologger)** OpenAI API logger to seamlessly log all all inputs and outputs to your OpenAI API. See the full Weights & Biases **[OpenAI Autologger Documentationhere](https://docs.wandb.ai/guides/integrations/openai)** for more

    ### Logging with just 1 line of code
    With just 1 line of code you can log all of the inputs and outputs from your OpenAI python libray to Weights & Biases for analysis later

    1️⃣. **Call autolog** and login to wandb with your wandb api key

    2️⃣. **Run OpenAI API** to generate preductions as normal

    3️⃣. **Visualize results** by doing to the wandb run link generated in step 1️⃣
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🪄 1. Install `wandb` and `openai`
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb openai !pip install wandb openai -qU
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🪄 2. Import and call `autolog`

    When you call `autolog` you will be prompted to login to Weights & Biases. If you haven't already, create a new API key at [wandb.ai/settings](https://wandb.ai/settings) and store it securely. API keys can only be viewed once when created.

    You can optionally pass a dictionary with arguments for [wandb.init()](https://docs.wandb.ai/ref/python/init) such as a project name, team name, entity, and more. For more information about wandb.init, see the [API Reference Guide](https://docs.wandb.ai/ref/python/init).
    """)
    return


@app.cell
def _():
    import openai
    from wandb.integration.openai import autolog

    autolog({"project":"my_llm_project"})
    return autolog, openai


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once you login, a **Weights & Biases run link will be generated**. This will take you to your workspace where you will be able to see all of the inputs and outputs to your OpenAI API calls.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🪄 3. Use the OpenAI API as normal
    """)
    return


@app.cell
def _(openai):
    # pass your OpenAI key
    openai.api_key = 'sk-foo'
    return


@app.cell
def _(openai):
    # make some calls to OpenAI 
    # Call 1
    chat_request_kwargs = dict(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the superbowl in 2014?"},
            {"role": "assistant", "content": "The Seattle Seahawks"},
            {"role": "user", "content": "Where was it played?"},
        ],
    )

    response_1 = openai.ChatCompletion.create(**chat_request_kwargs)
    print(response_1)

    # Call 2
    chat_request_kwargs = dict(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers"},
            {"role": "user", "content": "Where was it played?"},
        ],
    )

    response_2 = openai.ChatCompletion.create(**chat_request_kwargs)
    print(response_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🪄 4. View your logged results in Weights & Biases
    - You can find your interactive dashboard by clicking the 👆 wandb links above in step (2)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🪄 5. Disable `autolog`
    Call disable() to close all W&B processes when you are finished using the OpenAI API.
    """)
    return


@app.cell
def _(autolog):
    autolog.disable()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What's next 🚀 ?
    Doing more advanced LLM system chaining or using LangChain?
    ## 👉 [Try W&B Prompts to understand your LLM systems](https://docs.wandb.ai/guides/prompts?utm_source=openai_api_colab&utm_medium=code&utm_campaign=openai_api_colab)
    """)
    return


if __name__ == "__main__":
    app.run()
