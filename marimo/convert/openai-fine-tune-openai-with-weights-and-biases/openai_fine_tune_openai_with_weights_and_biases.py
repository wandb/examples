# /// script
# dependencies = ["datasets", "openai", "tenacity", "tiktoken", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/openai/Fine_tune_OpenAI_with_Weights_and_Biases.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{openai-finetune-gpt3} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{openai-finetune-gpt35} -->

    # Fine-tune ChatGPT-3.5 and GPT-4 with Weights & Biases
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you use OpenAI's API to [fine-tune ChatGPT-3.5](https://platform.openai.com/docs/guides/fine-tuning), you can now use the `WandbLogger` integration to track experiments, models, and datasets in your central dashboard with just two lines of code:

    ```
    from wandb.integration.openai.fine_tuning import WandbLogger

    # Your fine-tuning logic

    WandbLogger.sync(id=fine_tune_job_id)
    ```

    See the [OpenAI section](https://wandb.me/openai-docs) in the Weights & Biases documentation for full details of the integration.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb openai tiktoken datasets tenacity !pip install -Uq wandb openai tiktoken datasets tenacity
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this colab notebook, we will be finetuning GPT 3.5 model on the [LegalBench](https://hazyresearch.stanford.edu/legalbench/) dataset. The notebook will show how to prepare and validate the dataset, upload it to OpenAI and setup a fine-tune job. Finally, the notebook shows how to use the `WandbLogger`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Imports and initial set-up
    """)
    return


@app.cell
def _():
    from openai import OpenAI
    import wandb

    import os
    import glob
    import json
    import random
    import tiktoken
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from tqdm.auto import tqdm
    from collections import defaultdict
    from tenacity import retry, stop_after_attempt, wait_fixed

    return (
        OpenAI,
        defaultdict,
        glob,
        json,
        np,
        pd,
        random,
        tiktoken,
        tqdm,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Initialize the OpenAI client

    You can add the api key to your environment variable by doing `os.environ['OPENAI_API_KEY'] = "sk-...."`.
    """)
    return


@app.cell
def _(OpenAI):
    # Uncomment the line below and set your OpenAI API Key.
    # os.environ['OPENAI_API_KEY'] = "sk-...." 
    client = OpenAI()
    return (client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Import the `WandbLogger`
    """)
    return


@app.cell
def _():
    from wandb.integration.openai.fine_tuning import WandbLogger

    WANDB_PROJECT = "OpenAI-Fine-Tune"
    return WANDB_PROJECT, WandbLogger


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dataset Preparation

    We download a dataset from [LegalBench](https://hazyresearch.stanford.edu/legalbench/), a project to curate tasks for evaluating legal reasoning, specifically the [Contract NLI Explicit Identification task](https://github.com/HazyResearch/legalbench/tree/main/tasks/contract_nli_explicit_identification).

    This comprises of a total of 117 examples, from which we will create our own train and test datasets
    """)
    return


@app.cell
def _(random):
    from datasets import load_dataset
    dataset = load_dataset('nguha/legalbench', 'contract_nli_explicit_identification')
    # Download the data, merge into a single dataset and shuffle
    data = []
    for _d in dataset['train']:
        data.append(_d)
    for _d in dataset['test']:
        data.append(_d)
    random.shuffle(data)
    for _idx, _d in enumerate(data):
        _d['new_index'] = _idx
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's look at a few samples.
    """)
    return


@app.cell
def _(data):
    len(data), data[0:2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Format our Data for Chat Completion Models
    We modify the `base_prompt` from the LegalBench task to make it a zero-shot prompt, as we are training the model instead of using few-shot prompting
    """)
    return


@app.cell
def _():
    base_prompt_zero_shot = "Identify if the clause provides that all Confidential Information shall be expressly identified by the Disclosing Party. Answer with only `Yes` or `No`"
    return (base_prompt_zero_shot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now split it into training/validation dataset, lets train on 30 samples and test on the remainder
    """)
    return


@app.cell
def _(data):
    n_train = 30
    n_test = len(data) - n_train
    return n_test, n_train


@app.cell
def _(base_prompt_zero_shot, data, n_test, n_train):
    train_messages = []
    test_messages = []
    for _d in data:
        prompts = []
        prompts.append({'role': 'system', 'content': base_prompt_zero_shot})
        prompts.append({'role': 'user', 'content': _d['text']})
        prompts.append({'role': 'assistant', 'content': _d['answer']})
        if int(_d['new_index']) < n_train:
            train_messages.append({'messages': prompts})
        else:
            test_messages.append({'messages': prompts})
    (len(train_messages), len(test_messages), n_test, train_messages[5])
    return test_messages, train_messages


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save the data to Weights & Biases

    Save the data in a train and test file first
    """)
    return


@app.cell
def _(json, test_messages, train_messages):
    train_file_path = 'encoded_train_data.jsonl'
    with open(train_file_path, 'w') as _file:
        for item in train_messages:
            line = json.dumps(item)
            _file.write(line + '\n')
    test_file_path = 'encoded_test_data.jsonl'
    with open(test_file_path, 'w') as _file:
        for item in test_messages:
            line = json.dumps(item)
            _file.write(line + '\n')
    return test_file_path, train_file_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run the OpenAI data validation script
    Next, we validate that our training data is in the correct format using a script from the [OpenAI fine-tuning documentation](https://platform.openai.com/docs/guides/fine-tuning/)
    """)
    return


@app.cell
def _(defaultdict, json, np, tiktoken):
    def openai_validate_data(dataset_path):
        data_path = dataset_path
        with open(data_path) as f:
            dataset = [json.loads(line) for line in f]
        print('Num examples:', len(dataset))
        print('First example:')
        for message in dataset[0]['messages']:
            print(message)
        format_errors = defaultdict(int)
        for ex in dataset:
            if not isinstance(ex, dict):
                format_errors['data_type'] = format_errors['data_type'] + 1
                continue
            _messages = ex.get('messages', None)
            if not _messages:
                format_errors['missing_messages_list'] = format_errors['missing_messages_list'] + 1
                continue
            for message in _messages:
                if 'role' not in message or 'content' not in message:
                    format_errors['message_missing_key'] = format_errors['message_missing_key'] + 1
                if any((k not in ('role', 'content', 'name') for k in message)):
                    format_errors['message_unrecognized_key'] = format_errors['message_unrecognized_key'] + 1
                if message.get('role', None) not in ('system', 'user', 'assistant'):
                    format_errors['unrecognized_role'] = format_errors['unrecognized_role'] + 1
                content = message.get('content', None)
                if not content or not isinstance(content, str):
                    format_errors['missing_content'] = format_errors['missing_content'] + 1
            if not any((message.get('role', None) == 'assistant' for message in _messages)):
                format_errors['example_missing_assistant_message'] = format_errors['example_missing_assistant_message'] + 1
        if format_errors:
            print('Found errors:')
            for k, v in format_errors.items():
                print(f'{k}: {v}')
        else:
            print('No errors found')
        encoding = tiktoken.get_encoding('cl100k_base')

        def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
            num_tokens = 0
            for message in _messages:
                num_tokens = num_tokens + tokens_per_message
                for key, value in message.items():
                    num_tokens = num_tokens + len(encoding.encode(value))
                    if key == 'name':
                        num_tokens = num_tokens + tokens_per_name
            num_tokens = num_tokens + 3
            return num_tokens

        def num_assistant_tokens_from_messages(messages):
            num_tokens = 0
            for message in _messages:
                if message['role'] == 'assistant':
                    num_tokens = num_tokens + len(encoding.encode(message['content']))
            return num_tokens

        def print_distribution(values, name):
            print(f'\n#### Distribution of {name}:')
            print(f'min / max: {min(values)}, {max(values)}')
            print(f'mean / median: {np.mean(values)}, {np.median(values)}')
            print(f'p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}')
        n_missing_system = 0
        n_missing_user = 0
        n_messages = []
        convo_lens = []
        assistant_message_lens = []
        for ex in dataset:
            _messages = ex['messages']
            if not any((message['role'] == 'system' for message in _messages)):
                n_missing_system = n_missing_system + 1
            if not any((message['role'] == 'user' for message in _messages)):
                n_missing_user = n_missing_user + 1
            n_messages.append(len(_messages))
            convo_lens.append(num_tokens_from_messages(_messages))
            assistant_message_lens.append(num_assistant_tokens_from_messages(_messages))
        print('Num examples missing system message:', n_missing_system)
        print('Num examples missing user message:', n_missing_user)
        print_distribution(n_messages, 'num_messages_per_example')
        print_distribution(convo_lens, 'num_total_tokens_per_example')
        print_distribution(assistant_message_lens, 'num_assistant_tokens_per_example')
        n_too_long = sum((l > 4096 for l in convo_lens))
        print(f'\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning')
        MAX_TOKENS_PER_EXAMPLE = 4096
        MIN_TARGET_EXAMPLES = 100
        MAX_TARGET_EXAMPLES = 25000
        TARGET_EPOCHS = 3
        MIN_EPOCHS = 1
        MAX_EPOCHS = 25
        n_epochs = TARGET_EPOCHS
        n_train_examples = len(dataset)
        if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
            n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
        elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
            n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)
        n_billing_tokens_in_dataset = sum((min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens))
        print(f'Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training')
        print(f"By default, you'll train for {n_epochs} epochs on this dataset")
        print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
        print('See pricing page to estimate total costs')

    return (openai_validate_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Validate train data
    """)
    return


@app.cell
def _(openai_validate_data, train_file_path):
    openai_validate_data(train_file_path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Validate test data
    """)
    return


@app.cell
def _(openai_validate_data, test_file_path):
    openai_validate_data(test_file_path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Upload the training and validation data to OpenAI

    We will first upload the data to OpenAI. This might take a few minutes depending on the size of your dataset.
    """)
    return


@app.cell
def _(client, test_file_path, train_file_path):
    openai_train_file_info = client.files.create(
        file=open(train_file_path, "rb"), purpose="fine-tune"
    )

    openai_valid_file_info = client.files.create(
        file=open(test_file_path, "rb"), purpose="fine-tune"
    )
    return openai_train_file_info, openai_valid_file_info


@app.cell
def _(openai_train_file_info):
    openai_train_file_info
    return


@app.cell
def _(openai_valid_file_info):
    openai_valid_file_info
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Notice the unique ids for both training and validation data. OpenAI uses these ids to access the uploaded data to fine-tune GPT 3.5 on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train the model and log to Weights & Biases
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's define our ChatGPT-3.5 fine-tuning hyper-parameters.
    """)
    return


@app.cell
def _():
    model = 'gpt-3.5-turbo'
    n_epochs = 3
    return model, n_epochs


@app.cell
def _(client, model, n_epochs, openai_train_file_info, openai_valid_file_info):
    openai_ft_job_info = client.fine_tuning.jobs.create(
        training_file=openai_train_file_info.id,
        model=model,
        hyperparameters={"n_epochs": n_epochs},
        validation_file=openai_valid_file_info.id
    )

    ft_job_id = openai_ft_job_info.id
    return (ft_job_id,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > this takes around 5 minutes to train.

    ### Start Weight & Biases Sync
    Calling `WandbLogger.sync` will start polling OpenAI for the fine-tuning job results and log them when they are retrieved, see the [docs](https://docs.wandb.ai/guides/integrations/openai) for how to modify this behaviour
    """)
    return


@app.cell
def _(WANDB_PROJECT, WandbLogger, client, ft_job_id):
    # Log to Weights and Biases
    WandbLogger.sync(fine_tune_job_id=ft_job_id, project=WANDB_PROJECT, openai_client=client)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Thats it!**

    Now your model is training on OpenAI's machines.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Logging the fine-tuning job to W&B is straight forward. The integration will automatically log the following to W&B:

    - training and validation metrics (if validation data is provided)
    - log the training and validation data as W&B Tables for storage and versioning
    - log the fine-tuned model's metadata.

    The integration automatically creates the DAG lineage between the data and the model.

    > You can call the `WandbLogger` with the job id. The cell will keep running till the fine-tuning job is not complete. Once the job's status is `succeeded`, the `WandbLogger` will log metrics and data to W&B. This way you don't have to wait for the fine-tune job to be completed to call `WandbLogger.sync`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calling `WandbLogger.sync` without any id will log all un-synced fine-tuned jobs to W&B

    See the [OpenAI section](https://wandb.me/openai-docs) in the Weights & Biases documentation for full details of the integration
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The fine-tuning job is now successfully synced to Weights and Biases. Click on the URL above to open the [W&B run page](https://docs.wandb.ai/guides/app/pages/run-page). The following will be logged to W&B:

    #### Training and validation metrics

    ![image.png](assets/metrics.png)

    #### Training and validation data as W&B Tables

    ![image.png](assets/datatable.png)

    #### The data and model artifacts for version control (go to the overview tab)

    ![image.png](assets/artifacts.png)

    #### The configuration and hyperparameters (go to the overview tab)

    ![image.png](assets/configs.png)

    #### The data and model DAG

    ![image.png](assets/dag.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run evalution and log the results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The best way to evaluate a generative model is to explore sample predictions from your evaluation set.

    Let's generate a few inference samples and log them to W&B and see how the performance compares to a baseline ChatGPT-3.5 model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will be evaluating using the validation dataset. In the overview tab of the run page, find the "validation_files" in the Artifact Inputs section. Clicking on it will take you to the artifacts page. Copy the artifact URI (full name) as shown in the image below.

    ![image](assets/select_artifact_uri.png)
    """)
    return


@app.cell
def _(WANDB_PROJECT, wandb):
    run = wandb.init(
        project=WANDB_PROJECT,
        job_type='eval'
    )

    VALIDATION_FILE_ARTIFACT_URI = '<entity>/<project>/valid-file-*' # REPLACE THIS WITH YOUR OWN ARTIFACT URI

    artifact_valid = run.use_artifact(
        VALIDATION_FILE_ARTIFACT_URI,
        type='validation_files'
    )
    return artifact_valid, run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The code snippet below, download the logged validation data and prepare a pandas dataframe from it.
    """)
    return


@app.cell
def _(artifact_valid, glob, json, pd, run):
    artifact_valid_path = artifact_valid.download()
    print('Downloaded the validation data at: ', artifact_valid_path)
    validation_file = glob.glob(f'{artifact_valid_path}/*.table.json')[0]
    with open(validation_file, 'r') as _file:
        data_1 = json.load(_file)
    validation_df = pd.DataFrame(columns=data_1['columns'], data=data_1['data'])
    print(f'There are {len(validation_df)} validation examples')
    run.config.update({'num_validation_samples': len(validation_df)})
    validation_df.head()
    return (validation_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will need to package the data in the dataframe in the format acceptable by GPT 3.5. The format is:

    ```
    {"messages": [{"role": "system", "content": "some system prompt"}, {"role": "user", "content": "some user prompt"}, {"role": "assistant", "content": "completion text"}]}
    ```

    For evaluation we don't need to pack the `{"role": "assistant", "content": "completition text"}` in `messages` as this is meant to be generated by GPT 3.5.
    """)
    return


@app.cell
def _(validation_df):
    def eval_data_format(row):
        role_system_content = _row['role: system']
        role_system_dict = {'role': 'system', 'content': role_system_content}
        role_user_content = _row['role: user']
        role_user_dict = {'role': 'user', 'content': role_user_content}
        return [role_system_dict, role_user_dict]
    validation_df['messages'] = validation_df.apply(lambda row: eval_data_format(_row), axis=1)
    validation_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run evaluation on the Fine-Tuned Model

    Next up we will get the fine-tuned model's id from the logged `model_metadata`. In the overview tab of the run page, find the "model" in the Artifact Outputs section. Clicking on it will take you to the artifacts page. Copy the artifact URI (full name) as shown in the image below.

    ![image](assets/select_model_artifact.png)
    """)
    return


@app.cell
def _(run):
    MODEL_ARTIFACT_URI = '<entity>/<project>/model_metadata:v*' # REPLACE THIS WITH YOUR OWN ARTIFACT URI

    model_artifact = run.use_artifact(
        MODEL_ARTIFACT_URI,
        type='model'
    )
    return (model_artifact,)


@app.cell
def _(glob, json, model_artifact):
    model_metadata_path = model_artifact.download()
    print('Downloaded the validation data at: ', model_metadata_path)
    model_metadata_file = glob.glob(f'{model_metadata_path}/*.json')[0]
    with open(model_metadata_file, 'r') as _file:
        model_metadata = json.load(_file)
    model_metadata
    return (model_metadata,)


@app.cell
def _(OpenAI, model_metadata):
    fine_tuned_model = model_metadata['fine_tuned_model']
    client_1 = OpenAI()
    return client_1, fine_tuned_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run evaluation and log results to W&B
    """)
    return


@app.cell
def _(client_1, fine_tuned_model, tqdm, validation_df, wandb):
    prediction_table = wandb.Table(columns=['messages', 'completion', 'target'])
    eval_data = []
    for _idx, _row in tqdm(validation_df.iterrows()):
        _messages = _row.messages
        _target = _row['role: assistant']
        _res = client_1.chat.completions.create(model=fine_tuned_model, messages=_messages, max_tokens=10)
        _completion = _res.choices[0].message.content
        eval_data.append([_messages, _completion, _target])
        prediction_table.add_data(_messages[1]['content'], _completion, _target)
    wandb.log({'predictions': prediction_table})
    return (eval_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calculate the accuracy of the fine-tuned model and log to W&B
    """)
    return


@app.cell
def _(eval_data, wandb):
    correct = 0
    for _e in eval_data:
        if _e[1].lower() == _e[2].lower():
            correct = correct + 1
    accuracy = correct / len(eval_data)
    print(f'Accuracy is {accuracy}')
    wandb.log({'eval/accuracy': accuracy})
    wandb.summary['eval/accuracy'] = accuracy
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run evaluation on a Baseline model for comparison
    Lets compare our model to the baseline model, `gpt-3.5-turbo`
    """)
    return


@app.cell
def _(client_1, tqdm, validation_df, wandb):
    baseline_prediction_table = wandb.Table(columns=['messages', 'completion', 'target'])
    baseline_eval_data = []
    for _idx, _row in tqdm(validation_df.iterrows()):
        _messages = _row.messages
        _target = _row['role: assistant']
        _res = client_1.chat.completions.create(model='gpt-3.5-turbo', messages=_messages, max_tokens=10)
        _completion = _res.choices[0].message.content
        baseline_eval_data.append([_messages, _completion, _target])
        baseline_prediction_table.add_data(_messages[1]['content'], _completion, _target)
    wandb.log({'baseline_predictions': baseline_prediction_table})
    return (baseline_eval_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calculate the accuracy of the fine-tuned model and log to W&B
    """)
    return


@app.cell
def _(baseline_eval_data, wandb):
    baseline_correct = 0
    for _e in baseline_eval_data:
        if _e[1].lower() == _e[2].lower():
            baseline_correct = baseline_correct + 1
    baseline_accuracy = baseline_correct / len(baseline_eval_data)
    print(f'Baseline Accurcy is: {baseline_accuracy}')
    wandb.log({'eval/baseline_accuracy': baseline_accuracy})
    wandb.summary['eval/baseline_accuracy'] = baseline_accuracy
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And thats it! In this example we have prepared our data, logged it to Weights & Biases, fine-tuned an OpenAI model using that data, logged the results to Weights & Biases and then run evaluation on the fine-tuned model.

    From here you can start to train on larger or more complex tasks, or else explore other ways to modify ChatGPT-3.5 such as giving it a different tone and style or response.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Resources

    * [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
    * [W&B Integration with OpenAI API Documentation](https://wandb.me/openai-docs)
    """)
    return


if __name__ == "__main__":
    app.run()
