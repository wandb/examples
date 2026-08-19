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


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/prompts/prompts_evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{prompts-eval} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{prompts-eval} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Iterate and Evaluate LLM applications
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    AI application building is an experimental process where you likely don't know how a given system will perform on your task. To iterate on an application, we need a way to evaluate if it's improving. To do so, a common practice is to test it against the same dataset when there is a change.

    This tutorial will show you how to:
    - track input prompts and pipeline settings with `wandb.config`
    - track final evaluation metrics e.g. F1 score or scores from LLM judges, with `wandb.log`
    - track individual model predictions and metadata in `W&B Tables`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll track F1 score on extracting named entities from an example news headlines dataset from `explosion/prodigy-recipes` from the https://prodi.gy/ team.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    ## Download Data
    """)
    return


@app.cell
def _(subprocess):
    #! curl -O https://raw.githubusercontent.com/explosion/prodigy-recipes/master/example-datasets/annotated_news_headlines-ORG-PERSON-LOCATION-ner.jsonl
    subprocess.call(['curl', '-O', 'https://raw.githubusercontent.com/explosion/prodigy-recipes/master/example-datasets/annotated_news_headlines-ORG-PERSON-LOCATION-ner.jsonl'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Installation
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb openai !pip install wandb openai
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create a W&B account and log in
    """)
    return


@app.cell
def _():
    import wandb
    wandb.login()
    return (wandb,)


@app.cell
def _():
    import json
    from functools import partial
    import timeit
    import openai
    from concurrent.futures import ThreadPoolExecutor
    data = []
    with open('annotated_news_headlines-ORG-PERSON-LOCATION-ner.jsonl') as f:
        for line in f:
            data.append(json.loads(line))
    return ThreadPoolExecutor, data, partial, timeit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Format data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we just remove data we're not using and format the examples for our task.
    """)
    return


@app.cell
def _(data):
    def clean_examples():
        labelled_examples = []
        for example in data:
            entities = []
            if 'spans' in example:
                for span in example['spans']:
                    start = span['start']
                    end = span['end']
                    label = span['label']
                    text = ''  # Extract the corresponding text from tokens
                    for token in example['tokens']:
                        if token['start'] >= start and token['end'] <= end:
                            text = text + (token['text'] + ' ')
                    entities.append(text.rstrip())
            labelled_examples.append({'text': example['text'], 'entities': entities})
        return labelled_examples
    labelled_examples = clean_examples()
    return (labelled_examples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Set up LLM boilerplate
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll call `openai` (you'll need to add an OpenAI API key) with a given prompt to extract the entities and replace `<text>` with our input. We'll also grab useful metadata from the openai response for logging.
    """)
    return


@app.cell
def _(timeit):
    def extract_entities_with_template(text, template_prompt, system_prompt, model, temperature):
        start_time = timeit.default_timer()
        prompt = template_prompt.replace('<text>', text)
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}], temperature=temperature)
        text = response.choices[0].message.content
        entities = list(filter(None, text.split('\n')))
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        end_time = timeit.default_timer()
        _elapsed = end_time - start_time
        return {'entities': entities, 'model': model, 'prompt': prompt, 'elapsed': _elapsed, 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': total_tokens}

    return (extract_entities_with_template,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Calculate Metric
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we make an evaluation metric for our task.
    Note: It's not shown here, but you could also use an LLM to evaluate your task if it's not as straight forward to evaluate as this task.
    """)
    return


@app.function
def calculate_f1(extracted_entities, ground_truth_entities):
    extracted_set = set(map(str.lower, extracted_entities))
    ground_truth_set = set(map(str.lower, ground_truth_entities))
    tp_examples = extracted_set & ground_truth_set
    tp = len(tp_examples)
    fp_examples = extracted_set - ground_truth_set
    fp = len(fp_examples)
    fn_examples = ground_truth_set - extracted_set
    fn = len(fn_examples)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return f1, tp, fp, fn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Perform inference in parallel
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Running evaluations can be a bit slow. To speed it up, here is a bit of useful code to gather your examples in parallel. None of this is specific to W&B, but it's useful to have nonetheless.
    """)
    return


@app.cell
def _(
    ThreadPoolExecutor,
    extract_entities_with_template,
    labelled_examples,
    partial,
    timeit,
):
    def inference(examples, system_prompt, template_prompt, model, temperature):
        _extracted = []
        openai_func = partial(extract_entities_with_template, model=model, system_prompt=system_prompt, template_prompt=template_prompt, temperature=temperature)  # making a new function to openai which has the template
        start_time = timeit.default_timer()  # this is needed because exectutor.map wants a func with one arg
        with ThreadPoolExecutor(max_workers=8) as executor:
            for i in executor.map(openai_func, [t['text'] for t in examples]):
                _extracted.append(i)
        end_time = timeit.default_timer()  # Run the model to extract the entities
        _elapsed = end_time - start_time
        return (_extracted, _elapsed)
    model = 'gpt-3.5-turbo'
    temperature = 0.7
    template = '\ntext: <text>\nReturn the entities as a list with a new line between each entity.\n'
    system_prompt = 'You are an excellent entity extractor reading newspapers and extracting orgs, people and locations. Extract the entities from the follow sentence.'
    _extracted, _elapsed = inference(labelled_examples[:1], system_prompt, template, model, temperature)
    print(_extracted[0])
    print(labelled_examples[0]['text'])
    return inference, model, system_prompt, temperature, template


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluate extracted entities, save in W&B Table for inspection later
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we calcualte our metric across all of our predictions and log them to a `wandb.Table` for later inspection.
    """)
    return


@app.cell
def _(wandb):
    def evaluate(extracted, labelled_examples):
        total_tp, total_fp, total_fn = (0, 0, 0)
        eval_table = wandb.Table(columns=['pred', 'truth', 'f1', 'tp', 'fp', 'fn', 'prompt_tokens', 'completion_tokens', 'total_tokens'])
        for pred, gt in zip(_extracted, labelled_examples):
            f1, tp, fp, fn = calculate_f1(pred['entities'], gt['entities'])
            total_tp = total_tp + tp
            total_fp = total_fp + fp
            total_fn = total_fn + f1
            eval_table.add_data(pred['entities'], gt['entities'], f1, tp, fp, fn, pred['prompt_tokens'], pred['completion_tokens'], pred['total_tokens'])
        wandb.log({'eval_table': eval_table})
        _overall_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
        _overall_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
        _overall_f1 = 2 * _overall_precision * _overall_recall / (_overall_precision + _overall_recall) if _overall_precision + _overall_recall else 0
        return (_overall_precision, _overall_recall, _overall_f1)

    return (evaluate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run our pipeline:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To start logging to W&B, you can call `wandb.init` and pass in the config to track the configurations you're experimenting with currently.

    As you experiment, you can call `wandb.log` to track your work. This will log the metrics to W&B. Finally, we'll call `wandb.finish` to stop tracking. This will be tracked as one "Run" in W&B.

    You'll be given a link to W&B to see all of your logs.
    """)
    return


@app.cell
def _(
    evaluate,
    inference,
    labelled_examples,
    model,
    system_prompt,
    temperature,
    template,
    wandb,
):
    NUM_EXAMPLES = 50
    wandb.init(project='prompts_eval', config={'system_prompt': system_prompt, 'template': template, 'model': model, 'temperature': temperature})
    _extracted, _elapsed = inference(labelled_examples[:NUM_EXAMPLES], system_prompt, template, model, temperature)
    _overall_precision, _overall_recall, _overall_f1 = evaluate(_extracted, labelled_examples[:NUM_EXAMPLES])
    _total_tokens_sum = sum([pred['total_tokens'] for pred in _extracted])
    _completion_tokens_sum = sum([pred['completion_tokens'] for pred in _extracted])
    _prompt_tokens_sum = sum([pred['prompt_tokens'] for pred in _extracted])
    wandb.log({'precision': _overall_precision, 'recall': _overall_recall, 'f1': _overall_f1, 'time_elapsed_total': _elapsed, 'prompt_tokens': _prompt_tokens_sum, 'completion_tokens': _completion_tokens_sum, 'total_tokens': _total_tokens_sum})
    wandb.finish()
    return (NUM_EXAMPLES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Set up experiments

    Start a W&B run per experiment with `wandb.init`, store experiment details in `config` arg. Log results with `wandb.log`. Call `wandb.finish` to end experiment. Loop over all options in grid search to find best configuration.
    """)
    return


@app.cell
def _(NUM_EXAMPLES, evaluate, inference, labelled_examples, template, wandb):
    system_prompts = ['Extract the entities from the follow sentence.', 'You are an excellent entity extractor reading newspapers and extracting orgs, people and locations. Extract the entities from the follow sentence.']
    for system_prompt_1 in system_prompts:
        for temperature_1 in [0.2, 0.6, 0.9]:
            for model_1 in ['gpt-3.5-turbo', 'gpt-3.5-turbo-1106']:
                wandb.init(project='prompts_eval', config={'system_prompt': system_prompt_1, 'template': template, 'model': model_1, 'temperature': temperature_1})
                _extracted, _elapsed = inference(labelled_examples[:NUM_EXAMPLES], system_prompt_1, template, model_1, temperature_1)
                _overall_precision, _overall_recall, _overall_f1 = evaluate(_extracted, labelled_examples[:NUM_EXAMPLES])
                _total_tokens_sum = sum([pred['total_tokens'] for pred in _extracted])
                _completion_tokens_sum = sum([pred['completion_tokens'] for pred in _extracted])
                _prompt_tokens_sum = sum([pred['prompt_tokens'] for pred in _extracted])
                wandb.log({'precision': _overall_precision, 'recall': _overall_recall, 'f1': _overall_f1, 'time_elapsed_total': _elapsed, 'prompt_tokens': _prompt_tokens_sum, 'completion_tokens': _completion_tokens_sum, 'total_tokens': _total_tokens_sum})
                wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conclusion

    You've learned how to use W&B to track evaluations of your LLM applications.
    You've used `wandb.init` to start tracking, `wandb.log` to log summary evaluation metrics and `wandb.Table` to track individual predictions & scores.
    We've also shared some best practices to format your code to make it easier to run evaluations in parallel and track every iteration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trace your LLM application

    If you want to learn more and you're using complex pipelines of LLM calls, you can leverage W&B Prompts to view traces of your application and see inputs & ouputs of each LLM or function call.

    Learn more about W&B Prompts in the documentation here: [https://docs.wandb.ai/guides/prompts](https://docs.wandb.ai/guides/prompts)
    """)
    return


if __name__ == "__main__":
    app.run()
