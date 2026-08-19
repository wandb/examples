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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/gemini/How_to_use_Gemini_Pro_API_with_WB_Weave.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{gemini-weave-intro} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How to use Gemini Pro API with W&B Weave

    Read [our article](https://wandb.ai/prompt-eng/gemini-weave/reports/How-to-use-Gemini-Pro-API-with-W-B-Weave--Vmlldzo3NzEwNTA1) and follow along in this colab.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Installation
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !pip install google-generativeai weave -qqU
    return


@app.cell
def _():
    import google.generativeai as genai
    import weave

    return genai, weave


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set up your Google API key and log into W&B Weave

    To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.
    """)
    return


@app.cell
def _(genai):
    from google.colab import userdata
    GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    return


@app.cell
def _(weave):
    weave.init('prompt-eng/gemini-weave')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate a summary and track it in Weave
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !wget https://raw.githubusercontent.com/wandb/llm-workshop-fc2024/main/part_2_structured_outputs/longpaper.txt
    # with open('longpaper.txt', 'r') as file:
    #     long_paper_text = file.read()
    return


@app.cell
def _(genai):
    model_info = genai.get_model('models/gemini-1.5-pro-latest')
    print(model_info.input_token_limit)
    return


@app.cell
def _(genai, long_paper_text):
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    model.count_tokens(long_paper_text)
    return (model,)


@app.cell
def _(long_paper_text, model, weave):
    @weave.op()
    def generate_summary(text):
        prompt = "Generate a concise summary of below text:\n"
        response = model.generate_content(prompt + long_paper_text)
        return {
            'summary': response.text
        }

    return (generate_summary,)


@app.cell
def _(generate_summary, long_paper_text):
    summary = generate_summary(long_paper_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gemini API JSON Mode
    """)
    return


@app.cell
def _(genai):
    model_1 = genai.GenerativeModel('gemini-1.5-pro-latest', generation_config={'response_mime_type': 'application/json'})
    return (model_1,)


@app.cell
def _():
    from pydantic import BaseModel, Field

    class Summary(BaseModel):
        title: str
        summary: str = Field(description="plain short text summary without markdown")

    schema = Summary.model_json_schema()
    schema
    return (schema,)


@app.cell
def _():
    import json

    return (json,)


@app.cell
def _(json, model_1, weave):
    @weave.op()
    def create_prompt(text, schema):
        prompt = f'Generate a concise summary of below text using below JSON schema.\nPlease output plain text without markdown and limit it to 200 words.\nText:\n{text}\nJSON schema:\n{schema}\n'
        return prompt

    @weave.op()
    def generate_summary_1(text, schema):
        prompt = create_prompt(text, schema)
        response = model_1.generate_content(prompt)
        try:
            output = json.loads(response.text)
        except:
            output = response.text
        return {'summary': output}

    return (generate_summary_1,)


@app.cell
def _(generate_summary_1, long_paper_text, schema):
    new_summary = generate_summary_1(long_paper_text, schema)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation with Weave
    """)
    return


@app.cell
def _(genai, json, weave):
    from pydantic import model_validator
    import os
    import time

    os.environ['WEAVE_PARALLELISM'] = '1' # remove parallelism due to our Gemini quota, remove it if not needed


    class SummaryModel(weave.Model):
        model_name: str
        prompt_template: str
        json_schema: dict
        model: genai.GenerativeModel

        @model_validator(mode="before")
        def create_model(cls, v):
            model_name = v["model_name"]
            model = genai.GenerativeModel(model_name,
                    generation_config={"response_mime_type": "application/json"})
            v["model"] = model
            return v

        @weave.op()
        async def predict(self, text: str) -> dict:
            time.sleep(15) # remove if your Gemini quota allows for it :)
            prompt = self.prompt_template.format(text=text, schema=self.schema)
            response = self.model.generate_content(prompt)
            try:
                output = json.loads(response.text)
                return output[0]
            except:
                return {'summary': response.text}

    return (SummaryModel,)


@app.cell
def _():
    prompt_template = """Generate a concise summary of below text using below JSON schema.
    Please output plain text without markdown and limit it to 200 words.
    Text:
    {text}
    JSON schema:
    {schema}
    """
    return (prompt_template,)


@app.cell
def _(SummaryModel, prompt_template, schema):
    model_2 = SummaryModel(model_name='gemini-1.5-pro-latest', prompt_template=prompt_template, json_schema=schema)
    return (model_2,)


@app.cell
async def _(long_paper_text, model_2):
    await model_2.predict(long_paper_text)
    return


@app.cell
def _(weave):
    dataset_uri = "weave:///prompt-eng/gemini-weave/object/long_papers:9N9vkE4XY1SYoXLbvbCtP0YKqyqXErilG4XW8jYmQgE"
    dataset = weave.ref(dataset_uri).get()
    return (dataset,)


@app.cell
def _(weave):
    # Scoring function checking format adherence
    @weave.op()
    def check_formatting(model_output: dict) -> dict:
        # Check if length is smaller than threshold
        result = False
        if type(model_output) == list:
            model_output = model_output[0]
        if type(model_output) == dict:
            if 'summary' in model_output.keys():
                if type(model_output['summary']) == str:
                    result = True
        return {'formatting': result}

    return (check_formatting,)


@app.cell
def _(weave):
    # Scoring function checking length of summary
    @weave.op()
    def check_conciseness(model_output: dict) -> dict:
        # Check if length is smaller than threshold
        result = False
        if type(model_output) == list:
            model_output = model_output[0]
        if type(model_output) == dict:
            if 'summary' in model_output.keys():
                summary = model_output['summary']
                if type(summary) == str:
                    result = len(summary.split()) < 300
        return {'conciseness': result}

    return (check_conciseness,)


@app.cell
def _(check_conciseness, check_formatting, dataset, weave):
    evaluation = weave.Evaluation(
        dataset=dataset, scorers=[check_formatting, check_conciseness],
    )
    return (evaluation,)


@app.cell
async def _(evaluation, model_2):
    await evaluation.evaluate(model_2)
    return


if __name__ == "__main__":
    app.run()
