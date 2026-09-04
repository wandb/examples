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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/openai/Set_up_GPT_3_in_Python_with_the_OpenAI_API_and_Weights_&_Biases.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{gpt3_series_1} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/OpenAI_Logo.svg/2560px-OpenAI_Logo.svg.png" width="400" height="80" alt="OpenAI" />
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{gpt3_series_1} -->
    """)
    return


@app.cell
def _():
    import os
    os.environ['OPENAI_API_KEY'] = ''
    return (os,)


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # 
    # !pip install --upgrade openai wandb
    return


@app.cell
def _(os):
    import openai
    import wandb
    openai.api_key = os.getenv('OPENAI_API_KEY')
    return openai, wandb


@app.cell
def _(wandb):
    run = wandb.init(project='GPT-3 App in Python')
    prediction_table = wandb.Table(columns=["prompt", "completion"])
    return (prediction_table,)


@app.cell
def _(openai, prediction_table):
    gpt_prompt = "Correct this to standard English:\n\nShe no went to the market."


    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=gpt_prompt,
      temperature=0.5,
      max_tokens=256,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0,
    )


    print(response['choices'][0]['text'])


    prediction_table.add_data(gpt_prompt,response['choices'][0]['text'])
    return


@app.cell
def _(prediction_table, wandb):
    wandb.log({'predictions': prediction_table})
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
