#!/usr/bin/env python
#
# This file is a modified version of the Hugging Face example training scipt
# which can be found here: https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification

# Imports
import os
import logging
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_from_disk, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# ✍️ import W&B ✍️
import wandb

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.10.0.dev0")
require_version("datasets>=1.8.0", "Please upgrade dataset; pip install datasets --upgrade")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on"},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        pass


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ✍️ Create a new run in to Weights & Biases and set the project name ✍️
    project_name = "hf-sagemaker"
    job_type=None
    if training_args.run_name == 'tmp':
        name = f"{model_args.model_name_or_path}_{training_args.learning_rate}_{training_args.warmup_steps}"
    elif "hpt" in training_args.run_name:
        name = f"HypTn_{model_args.model_name_or_path}_{training_args.learning_rate}_{training_args.warmup_steps}"
        job_type='HyperparameterTuning'
    else:
        name = training_args.run_name
        
    wandb.init(name=name, project=project_name, job_type=job_type)
    os.environ["WANDB_LOG_MODEL"] = "TRUE"  # Hugging Face Trainer will use this to log model weights to W&B
        
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name == 'banking77_artifacts':
        
        # Download the tokenized Datasets from W&B Artifacts and load to HF Datasets object
        for split in ['train', 'eval']:
            pth = f'./{split}'
            nm = f"{split}_{model_args.model_name_or_path.split('/')[-1]}_tokenized"
            artifact = wandb.use_artifact(f'morgan/hf-sagemaker/{nm}:v0', type=f'{split}_tokenized_dataset')
            artifact_dir = artifact.download(pth)
            if split == 'train':
                train_dataset = load_from_disk(pth)
            else:
                eval_dataset = load_from_disk(pth)
        
    elif data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        raise ValueError(
                f"dataset_name must be passed"
            )

    # Labels
    is_regression = False
    if data_args.dataset_name == 'banking77_artifacts':
        label_list = train_dataset.features["label"].names
    else: 
        label_list = raw_datasets["train"].features["label"].names
    
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        
    
    # Map labels to ids
    label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples['text'], padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if "label" in examples:
            result["label"] = examples["label"]
        return result
    
    if data_args.dataset_name != 'banking77_artifacts':
        with training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        if training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            if "validation" not in raw_datasets and "validation_matched" not in raw_datasets and "test" in raw_datasets:
                raw_datasets['validation'] = raw_datasets['test']
            elif "validation" not in raw_datasets and "validation_matched" not in raw_datasets and "test" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
            if data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

                
#     # ✍️ Log the training and eval datasets as a Weights & Biases Tables to Artifacts ✍️
#     for d_idx, ds in enumerate([train_dataset, eval_dataset]):
        
#         # Create W&B Table
#         dataset_table = wandb.Table(columns=['id', 'label_id', 'label', 'text'])
        
#         # Ensure different row ids when logging train and eval data
#         if d_idx == 1:
#             idx_step = len(train_dataset)
#             nm = 'eval'
#         else:
#             idx_step = 0
#             nm = 'train'
          
#         # Add each row of data to the table
#         for index in range(len(ds)):
#             idx = index + idx_step
                
#             lbl = ds[index]['label']
#             row = [idx,                    
#                    lbl, 
#                    model.config.id2label[lbl],                
#                    ds[index]['text']
#                   ]
#             dataset_table.add_data(*row)
        
#         # Log the table to Weights & Biases
#         dataset_artifact = wandb.Artifact(f"{data_args.dataset_name}_{nm}_dataset", type=f"{nm}_dataset")
#         dataset_artifact.add(dataset_table, f"{data_args.dataset_name}_{nm}")
#         wandb.log_artifact(dataset_artifact)   
            
    
    # Get the metric function
    metric = load_metric("accuracy")

    class ComputeMetrics:
        def __init__(self, train_len, eval_steps, log_predictions=False):
            self.train_len = train_len
            self.eval_steps = eval_steps
            self.log_predictions = log_predictions
            self.eval_step_count = eval_steps

        def __call__(self, p: EvalPrediction):
                preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
                preds_idxs = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
                preds_vals = np.max(preds, axis=1)
                
                # Create W&B Table
                validation_table = wandb.Table(columns=['id', 'step', 'pred_label_id', 'pred_score'])
                
                if self.log_predictions:
                    # Add predictions to your table
                    for i, val_pred in enumerate(preds_idxs):
                        idx = i + len(train_dataset)
                        row = [idx, self.eval_step_count, val_pred, preds_vals[i]]                    
                        validation_table.add_data(*row)

                    wandb.log({f'eval_predictions_table/{data_args.dataset_name}_preds_step_{self.eval_step_count}' : 
                               validation_table}, commit=False)
                    # increment step count
                    self.eval_step_count+=self.eval_steps
                
                return {"accuracy": (preds_idxs == p.label_ids).astype(np.float32).mean().item()}
        
    compute_metrics = ComputeMetrics(len(train_dataset), training_args.eval_steps, False)

    
    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

    wandb.finish()

    
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
