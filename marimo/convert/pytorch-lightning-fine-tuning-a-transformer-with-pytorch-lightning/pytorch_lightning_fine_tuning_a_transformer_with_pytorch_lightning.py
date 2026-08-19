# /// script
# dependencies = ["lightning", "pandas", "torch", "transformers", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Fine_tuning_a_Transformer_with_Pytorch_Lightning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{lightning_hf} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{lightning_hf} -->

    # Train a Model to Check Your Grammar Using W&B, PyTorch Lightning ⚡, and 🤗

    *Based on Ayush Chaurasia's awesome [W&B report](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA) and [colab](https://colab.research.google.com/drive/1SQ-FOgji8AiyrQ08sIVfDiA8OUw4bC12?usp=sharing) which performs the same task using BERT, vanilla PyTorch, and W&B.*

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook, we are going to train a model to detect ungrammatical sentences from the CoLA dataset. To perform the classification, we will be using Pytorch Lightning ⚡ to fine tune [DistilBERT](https://arxiv.org/abs/1910.01108), a transformer model from huggingface 🤗.

    We'll use Weights & Biases to:
    - Version our model inputs and outputs using [W&B Artifacts](https://docs.wandb.ai/guides/artifacts), including preprocessing steps, train/validation splits, and model checkpoints
    - Log and visualize training and validation performance using [W&B's Pytorch Lightning integration](https://docs.wandb.ai/guides/integrations/lightning)
    - Visualize and explore the raw dataset using [W&B Tables](https://docs.wandb.ai/guides/data-vis)
    - Orchestrate a hyperparameter search using [W&B Sweeps](https://docs.wandb.ai/guides/sweeps)

    Be sure to follow the links that each run outputs to your W&B workspace, where you will be able to see...

    **Your model's performance metrics updating in real time**

    ![](https://i.imgur.com/8yejscO.png)

    **The raw data as a W&B Table, which you can sort, group, and filter**

    ![](https://imgur.com/oiQ8RE4.png)

    **An awesome artifact graph showing our full pipeline**

    ![](https://imgur.com/vMJqKw7.png)

    **Interactive visualizations of how our hyperparameter choices effect model performance**

    ![](https://imgur.com/Twq7V6c.png)
    """)
    return


@app.cell
def _():
    # Install some dependencies
    # packages added via marimo's package management: pandas torch lightning transformers !pip install pandas torch lightning transformers
    # packages added via marimo's package management: wandb !pip install -Uq wandb
    return


@app.cell
def _():
    # Bulk import cell
    import wandb
    import random
    import torch
    import transformers
    import numpy as np
    import pandas as pd
    import lightning.pytorch as pl

    return pd, pl, torch, transformers, wandb


@app.cell
def _(pl):
    # Derandomizing cell
    pl.seed_everything(1234)
    return


@app.cell
def _():
    """
    Note that if you are using W&B local you will need to pass the url of your W&B 

    For example:
    """
    return


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell
def _():
    project = "grammar-checker"  # W&B project name here
    entity = None  # your W&B username or teamname here
    return entity, project


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The CoLA Dataset 🥤

    We’ll fine tune the model on The Corpus of Linguistic Acceptability (CoLA) dataset for single sentence classification. It’s a set of sentences labeled as grammatically correct or incorrect. It was first published in May of 2018, and is one of the tests included in the “GLUE Benchmark” on which models like DistilBERT are competing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll use a [reference artifact](https://docs.wandb.ai/guides/artifacts/references) to store a pointer to the source data. The advantages of doing this are:
    * Any runs that use this artifact reference will be able to trace their lineage back to the true source
    * We can use W&B to download the raw data in our code.

    The cell below starts a run with job type `register-data`. In the context of this run, we:
     1. Create an artifact called `cola-raw`
     2. Add a reference to the CoLA dataset to our `cola-raw` artifact
     3. Log the `cola-raw` artifact to Weights & Biases.
    """)
    return


@app.cell
def _(entity, project, wandb):
    # Enter the context of a W&B Run object, referenceable with the 'run' variable
    with wandb.init(entity=entity, project=project, job_type='register-data') as _run:
        data_source = wandb.Artifact('cola-raw', type='dataset')
        data_source.add_reference('https://nyu-mll.github.io/CoLA/cola_public_1.1.zip', name='zipfile')  # Construct a wandb.Artifact object
        _run.log_artifact(data_source)  # Store a reference to the download URL of the CoLA dataset  # Log the artifact to W&B
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tokenization 🪙

    The cell below defines the function `tokenize_data`, which transforms a list of sentences and a list of labels into a tuple of `torch.tensor` objects which can be consumed by the transormer model we'll be using. The 3 tensors returned are the tokenized form of the sentences, the attention masks indicating which tokens in each sentence correspond to actual words, and a tensor containing the original labels.
    """)
    return


@app.cell
def _(torch, transformers):
    def tokenize_data(sentences, labels):

      # Tokenize all of the sentences and map the tokens to thier word IDs.
      input_ids = []
      attention_masks = []

      # Get BertTokenizer from transformers
      tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

      # For every sentence...
      for sent in sentences:
    
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 64,           # Pad & truncate all sentences.
                          padding='max_length',
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',     # Return pytorch tensors.
                       )
    
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
    
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

      # Convert the lists into tensors.
      input_ids = torch.cat(input_ids, dim=0)
      attention_masks = torch.cat(attention_masks, dim=0)
      labels = torch.tensor(labels)
      return input_ids, attention_masks, labels

    return (tokenize_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The code below executes a run of type `preprocess-data`, which will
    1. Download the CoLA dataset using the reference artifact we logged previously
    2. Log the entire dataset to W&B as a Table
    3. Use the function `tokenize_data` to transform each sentence into a sequence of tokens and an attention mask
    4. Log the preprocessed data as an artifact to W&B.
    """)
    return


@app.cell
def _(entity, pd, project, subprocess, tokenize_data, torch, wandb):
    with wandb.init(entity=entity, project=project, job_type='preprocess-data') as _run:
        raw_data_artifact = _run.use_artifact('cola-raw:latest')
        zip_path = raw_data_artifact.get_entry('zipfile').download()  # Download the raw cola data from the 'zipfile' reference we added to the cola-raw artifact.
        subprocess.call(['unzip', '-o', '$zip_path'])
        df = pd.read_csv('./cola_public/raw/in_domain_train.tsv', delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
        _run.log({'raw-data': wandb.Table(dataframe=df)})  #! unzip -o $zip_path
        input_ids, attention_masks, labels = tokenize_data(df.sentence.values, df.label.values)
        preprocessed_data = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)  # jupyter hack to unzip data :P
        data_artifact = wandb.Artifact('preprocessed-data', type='dataset')
        with open('preprocessed-data.pt', 'wb') as f:  # Read in the raw data, log it to W&B as a wandb.Table
            torch.save(preprocessed_data, f)
        data_artifact.add_file('preprocessed-data.pt', name='dataset')
        _run.log_artifact(data_artifact)  # Perform tokenization and store as a TensorDataset  # 1. Create an artifact called preprocessed-data  # 2. Save the dataset to a local fil called preprocessed-data.pt  # 3. Add that file to the preprocessed-data artifact  # 4. Log the artifact to W&B
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Splitting Our Data 🪓

    For our training process, we want to split the data into a train and validation set. The train set is the data we will use to update the model parameters, while the validation set will be a smaller segment of data that we use to test whether our model is generalizing to examples that it hasn't been trained on.

    The cell below executes a `wandb.Run` with `job_type="split-data"`. In the context of this run we will:

    1. Download the `preprocessed-data` artifact logged by our previous run
    2. Use the `random_split` function from `torch` to perform a randomn 90/10 test/valiation split on the preprocessed data
    3. Store the split datasets in a new artifact called `split-dataset`
    """)
    return


@app.cell
def _(entity, project, torch, wandb):
    with wandb.init(entity=entity, project=project, job_type='split-data') as _run:
        pp_data_artifact = _run.use_artifact('preprocessed-data:latest')
        data_path = pp_data_artifact.get_entry('dataset').download()  # Download the preprocessed data
        dataset = torch.load(data_path, weights_only=False)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        split_data_artifact = wandb.Artifact('split-dataset', type='dataset')  # Calculate the number of samples to include in each set.
        torch.save(train_dataset, 'train.pt')
        torch.save(val_dataset, 'validation.pt')
        split_data_artifact.add_file('train.pt', name='train-data')
        split_data_artifact.add_file('validation.pt', name='validation-data')  # Divide the dataset by randomly selecting samples.
        _run.log_artifact(split_data_artifact)  # Construct a new artifact  # Save the dataset splits to disk  # Add the data splits to the artifact  # Log the artifact to W&B
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining Our Model ⚡

    We define our model and the associated training + validation procedures in the `LightningModule` below. The model itself is a pre-trained `DistilBertForSequenceClassification` with two labels.
    """)
    return


@app.cell
def _():
    from torch.optim import AdamW

    return (AdamW,)


@app.cell
def _(AdamW, pl, torch, transformers):
    class SentenceClassifier(pl.LightningModule):
  
      def __init__(self, learning_rate=5e-5):
        super(SentenceClassifier, self).__init__()
    
        # Load pretrained distilbert-base-uncased configured for classification with 2 labels
        self.model = transformers.DistilBertForSequenceClassification.from_pretrained(
          "distilbert-base-uncased", 
          num_labels = 2, 
          output_attentions = False, # Whether the model returns attentions weights.
          output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        self.learning_rate = learning_rate

      def training_step(self, batch, batch_no):
        """
        This function overrides the pl.LightningModule class. 
    
        When trainer.fit is called, each batch from the provided data loader is fed 
        to this function successively. 
        """
        ids, masks, labels = batch
        outputs = self.model(ids, attention_mask=masks, labels=labels)
        preds = torch.argmax(outputs["logits"], axis=1)
        correct = sum(preds.flatten() == labels.flatten())
        self.log("train/loss", outputs["loss"], on_step=True, on_epoch=True)
        self.log("train/acc", correct/len(ids), on_step=True, on_epoch=True)
        return outputs["loss"]

      def validation_step(self, batch, batch_no):
        """
        """
        ids, masks, labels = batch
        outputs = self.model(ids, attention_mask=masks, labels=labels)
        preds = torch.argmax(outputs["logits"], axis=1)
        correct = sum(preds.flatten() == labels.flatten())
        self.log("val/loss", outputs["loss"], on_step=False, on_epoch=True)
        self.log("val/acc", correct/len(ids), on_step=False, on_epoch=True)

      def configure_optimizers(self):
        return AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=1e-8
        )

    return (SentenceClassifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training & Tracking Our Model 📉

    In the cell below, we define a function `train` which sets up and performs training in the context of a W&B run. The train function takes a configuration dictionary as input then passes it to `wandb.init` via the `config` keyword argument. We use the values saved in the `wandb.config` object associated with the run to set the parameters of our trainer and data loaders. This is a crucial best practice to ensure that the values logged in the `config` object (and displayed in the run table of the W&B app) represent the actual parameters of the experiment.
    """)
    return


@app.cell
def _(SentenceClassifier, entity, pl, project, torch, wandb):
    def train(config={'learning_rate': 5e-05, 'batch_size': 16, 'epochs': 2}):
        with wandb.init(project=project, entity=entity, job_type='train', config=config) as _run:
            data = _run.use_artifact('split-dataset:latest')
            train_dataset = torch.load(data.get_entry('train-data').download(), weights_only=False)
            val_dataset = torch.load(data.get_entry('validation-data').download(), weights_only=False)  # Load the datasets from the split-dataset artifact
            config = _run.config
            model = SentenceClassifier(learning_rate=config.learning_rate)
            logger = pl.loggers.WandbLogger(experiment=_run, log_model=True)
            gpus = -1 if torch.cuda.is_available() else 0
            trainer = pl.Trainer(max_epochs=config.epochs, logger=logger)  # Extract the config object associated with the run
            train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size)
            val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)
            trainer.fit(model, train_data_loader, val_data_loader)  # Construct our LightningModule with the learning rate from the config object  # This logger is used when we call self.log inside the LightningModule  # Use as many GPUs as are available  # Construct a Trainer object with the W&B logger we created and epoch set by the config object  # Build data loaders for our datasets, using the batch_size from our config object  # Execute training

    return (train,)


@app.cell
def _(train):
    train()  # Run training with default parameters
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Running a Hyperparameter Sweep 🧹

    W&B sweeps allow you to optimize your model hyperparameters with minimal effort. In general, the workflow of sweeps is:
    1. Construct a dictionary or YAML file that defines the hyperparameter space
    2. Call `wandb.sweep(<sweep-dict>)` from the python library or `wandb sweep <yaml-file>` from the command line to initialize the sweep in W&B
    3. Run `wandb.agent(<sweep-id>)` (python lib) or `wandb agent <sweep-id>` (cli) to start a sweep agent to continuously:
      - pull hyperparameter combinations from W&B
      - run training with the given hyperparameters
      - log training metrics back to W&B

    <img src="https://i.imgur.com/zlbw3vQ.png" alt="sweeps-diagram" width="500">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We implement the sweeps workflow laid out above by:
    1. Creating a `sweep_config` dictionary describing our hyperparameter space and objective
      - The hyperparameters we will sweep over are `learning_rate`, `batch_size`, and `epochs`
      - Our objective in this sweep is to maximize the `validation/epoch_acc` metric logged to W&B
      - We will use the `random` strategy, which means we will sample uniformly from the parameter space indefinitely
    2. Calling `wandb.sweep(sweep_config)` to create the sweep in our W&B project
      - `wandb.sweep` will return a unique id for the sweep, saved as `sweep_id`
    3. Calling `wandb.agent(sweep_id, function=train)` to create an agent that will execute training with different hyperparameter combinations
      - The agent will repeatedly query W&B for hyperparameter combinations
      - When `wandb.init` is called within an agent, the `config` dictionary of the returned `run` will be populated with the next hyperparameter combination in the sweep
    """)
    return


@app.cell
def _():
    sweep_config = {
        'method': 'random',  # Randomly sample the hyperparameter space (alternatives: grid, bayes)
        'metric': {  # This is the metric we are interested in maximizing
          'name': 'validation/epoch_acc',
          'goal': 'maximize'   
        },
        # Paramters and parameter values we are sweeping across
        'parameters': {
            'learning_rate': {
                'values': [5e-5, 3e-5, 2e-5]
            },
            'batch_size': {
                'values': [16, 32]
            },
            'epochs':{
                'values': [1, 2]
            }
        }
    }
    return (sweep_config,)


@app.cell
def _(entity, project, sweep_config, wandb):
    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    return (sweep_id,)


@app.cell
def _(sweep_id, train, wandb):
    # Run an agent 🕵️ to try out 5 hyperparameter combinations
    wandb.agent(sweep_id, function=train, count=5)
    return


if __name__ == "__main__":
    app.run()
