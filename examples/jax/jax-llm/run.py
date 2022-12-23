import json
import logging
import math
import os
import sys
import time
from itertools import chain

# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from datasets import load_dataset, load_from_disk
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from huggingface_hub import Repository
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForMaskedLM,
    HfArgumentParser,
    is_tensorboard_available,
    set_seed,
)
from transformers.utils import get_full_repo_name, send_example_telemetry

from args import DataTrainingArguments, ModelArguments, TrainingArguments
from data import FlaxDataCollatorForLanguageModeling
from utils import generate_batch_splits


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_mlm", model_args, data_args, framework="flax")

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
        datefmt="[%X]",
    )

    logger = logging.getLogger(__name__)

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name,
                token=training_args.hub_token,
            )
        else:
            repo_name = training_args.hub_model_id
        repo = Repository(training_args.output_dir, clone_from=repo_name)

    wandb.init(project="protobert")
    wandb.config.update(training_args)
    wandb.config.update(model_args)
    wandb.config.update(data_args)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the wandb artifacts.
        if data_args.dataset_artifact_path is None:
            datasets = load_from_disk(data_args.dataset_name)
        else:
            artifact = wandb.use_artifact(data_args.dataset_artifact_path)
            local_path = artifact.download()

            datasets = load_from_disk(os.path.join(local_path, data_args.dataset_name))
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_artifact_path:
        artifact = wandb.use_artifact(model_args.tokenizer_artifact_path)
        local_path = artifact.download()
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(local_path, model_args.tokenizer_name)
        )
    elif model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = data_args.max_seq_length

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples = [
                line for line in examples if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples,
                return_special_tokens_mask=True,
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            input_columns=[text_column_name],
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    else:

        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [
                    t[i : i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)
                ]
                for k, t in concatenated_examples.items()
            }
            return result

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    encoded = tokenized_datasets["train"][0]["input_ids"]
    decoded = tokenizer.decode(encoded)
    logger.info(f"Tokenized first example encoded: {encoded}")
    logger.info(f"Tokenized first example decoded: {decoded}")

    tpu_count = jax.device_count(backend="gpu")
    logger.info(f"TPU devices: {tpu_count}")

    data_collator = FlaxDataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=data_args.mlm_probability
    )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    if model_args.model_name_or_path:
        model = FlaxAutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = FlaxAutoModelForMaskedLM.from_config(
            config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
        )

    if training_args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = (
        int(training_args.per_device_train_batch_size) * jax.device_count()
    )
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()

    num_train_steps = len(tokenized_datasets["train"]) // train_batch_size * num_epochs

    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=training_args.learning_rate,
        transition_steps=training_args.warmup_steps,
    )
    decay_fn = optax.linear_schedule(
        init_value=training_args.learning_rate,
        end_value=0,
        transition_steps=num_train_steps - training_args.warmup_steps,
    )
    linear_decay_lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[training_args.warmup_steps]
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = set(
            [
                layer[-2:]
                for layer_norm_name in layer_norm_candidates
                for layer in flat_params.keys()
                if layer_norm_name in "".join(layer).lower()
            ]
        )
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params)
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    if training_args.adafactor:
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=linear_decay_lr_schedule_fn,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=linear_decay_lr_schedule_fn,
            b1=training_args.adam_beta1,
            b2=training_args.adam_beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            mask=decay_mask_fn,
        )

    # Setup train state
    state = train_state.TrainState.create(
        apply_fn=model.__call__, params=model.params, tx=optimizer
    )

    # Define gradient update step fn
    def train_step(state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def loss_fn(params):
            labels = batch.pop("labels")

            logits = state.apply_fn(
                **batch, params=params, dropout_rng=dropout_rng, train=True
            )[0]

            # compute loss, ignore padded input tokens
            label_mask = jnp.where(labels > 0, 1.0, 0.0)
            loss = (
                optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
                * label_mask
            )

            # take average
            loss = loss.sum() / label_mask.sum()

            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)

        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)},
            axis_name="batch",
        )

        return new_state, metrics, new_dropout_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")

        logits = model(**batch, params=params, train=False)[0]

        # compute loss, ignore padded input tokens
        label_mask = jnp.where(labels > 0, 1.0, 0.0)
        loss = (
            optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
            * label_mask
        )

        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels) * label_mask

        # summarize metrics
        metrics = {
            "loss": loss.sum(),
            "accuracy": accuracy.sum(),
            "normalizer": label_mask.sum(),
        }
        metrics = jax.lax.psum(metrics, axis_name="batch")

        return metrics

    p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = jax_utils.replicate(state)

    train_time = 0
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    logger.info("Started training!")
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()
        train_metrics = []

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # Generate an epoch by shuffling sampling indices from the train dataset
        num_train_samples = len(tokenized_datasets["train"])
        logger.info(f"Num train samples: {num_train_samples}")
        # Avoid using jax.numpy here in case of TPU training
        train_samples_idx = np.random.permutation(np.arange(num_train_samples))
        train_batch_idx = generate_batch_splits(train_samples_idx, train_batch_size)

        # Gather the indexes for creating the batch and do a training step
        for step, batch_idx in enumerate(
            tqdm(train_batch_idx, desc="Training...", position=1)
        ):
            samples = [tokenized_datasets["train"][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples, pad_to_multiple_of=16)

            # Model forward
            model_inputs = shard(model_inputs.data)
            state, train_metric, dropout_rngs = p_train_step(
                state, model_inputs, dropout_rngs
            )
            train_metrics.append(train_metric)

            cur_step = epoch * (num_train_samples // train_batch_size) + step

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                # Save metrics
                train_metric = jax_utils.unreplicate(train_metric)
                train_time += time.time() - train_start
                if jax.process_index() == 0:
                    print(train_metric)
                    wandb.log({"train/loss": train_metric["loss"].item()})
                    wandb.log({"learning_rate": train_metric["learning_rate"].item()})

                epochs.write(
                    f"Step... ({cur_step} | Loss: {train_metric['loss']}, Learning Rate:"
                    f" {train_metric['learning_rate']})"
                )

                train_metrics = []

            if cur_step % training_args.eval_steps == 0 and cur_step > 0:
                # ======================== Evaluating ==============================
                num_eval_samples = len(tokenized_datasets["validation"])
                # Avoid using jax.numpy here in case of TPU training
                eval_samples_idx = np.arange(num_eval_samples)
                eval_batch_idx = generate_batch_splits(
                    eval_samples_idx, eval_batch_size, drop_last=False
                )

                eval_metrics = []
                for i, batch_idx in enumerate(
                    tqdm(eval_batch_idx, desc="Evaluating ...", position=2)
                ):
                    samples = [
                        tokenized_datasets["validation"][int(idx)] for idx in batch_idx
                    ]
                    model_inputs = data_collator(samples, pad_to_multiple_of=16)

                    # Model forward
                    metrics = pad_shard_unpad(p_eval_step, static_return=True)(
                        state.params,
                        model_inputs.data,
                        min_device_batch=per_device_eval_batch_size,
                    )
                    eval_metrics.append(metrics)

                # normalize eval metrics
                eval_metrics = get_metrics(eval_metrics)
                eval_metrics = jax.tree_util.tree_map(jnp.sum, eval_metrics)
                eval_normalizer = eval_metrics.pop("normalizer")
                eval_metrics = jax.tree_util.tree_map(
                    lambda x: x / eval_normalizer, eval_metrics
                )

                # Update progress bar
                epochs.desc = f"Step... ({cur_step} | Loss: {eval_metrics['loss']}, Acc: {eval_metrics['accuracy']})"

                # Save metrics
                if jax.process_index() == 0:
                    wandb.log({"eval/loss": eval_metrics["loss"].item()})
                    wandb.log({"eval/accuracy": eval_metrics["accuracy"].item()})

            if cur_step % training_args.save_steps == 0 and cur_step > 0:
                # save checkpoint after each epoch and push checkpoint to the hub
                if jax.process_index() == 0:
                    params = jax.device_get(
                        jax.tree_util.tree_map(lambda x: x[0], state.params)
                    )
                    model.save_pretrained(training_args.output_dir, params=params)
                    tokenizer.save_pretrained(training_args.output_dir)
                    if training_args.push_to_hub:
                        repo.push_to_hub(
                            commit_message=f"Saving weights and logs of step {cur_step}",
                            blocking=False,
                        )

    # Eval after training
    if training_args.do_eval:
        num_eval_samples = len(tokenized_datasets["validation"])
        # Avoid using jax.numpy here in case of TPU training
        eval_samples_idx = np.arange(num_eval_samples)
        eval_batch_idx = generate_batch_splits(
            eval_samples_idx, eval_batch_size, drop_last=False
        )

        eval_metrics = []
        for _, batch_idx in enumerate(
            tqdm(eval_batch_idx, desc="Evaluating ...", position=2)
        ):
            samples = [tokenized_datasets["validation"][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples, pad_to_multiple_of=16)

            # Model forward
            metrics = pad_shard_unpad(p_eval_step, static_return=True)(
                state.params,
                model_inputs.data,
                min_device_batch=per_device_eval_batch_size,
            )
            eval_metrics.append(metrics)

        # normalize eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_util.tree_map(
            lambda metric: jnp.sum(metric).item(), eval_metrics
        )
        eval_normalizer = eval_metrics.pop("normalizer")
        eval_metrics = jax.tree_util.tree_map(
            lambda x: x / eval_normalizer, eval_metrics
        )

        try:
            perplexity = math.exp(eval_metrics["loss"])
        except OverflowError:
            perplexity = float("inf")
        eval_metrics["perplexity"] = perplexity

        if jax.process_index() == 0:
            eval_metrics = {
                f"eval_{metric_name}": value
                for metric_name, value in eval_metrics.items()
            }
            path = os.path.join(training_args.output_dir, "eval_results.json")
            with open(path, "w") as f:
                json.dump(eval_metrics, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
