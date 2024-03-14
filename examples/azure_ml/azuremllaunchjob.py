# !pip install transformers datasets "transformers[torch]" wandb
# !pip install onnx onnxruntime
# !pip install tensorflow
# !pip install -U accelerate

import os
os.environ["WANDB_PROJECT"] = "azure_ml_test"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, DatasetDict
import wandb
import numpy as np
import pandas as pd
from transformers.integrations import WandbCallback
from pathlib import Path
from transformers import TFAutoModelForSequenceClassification
from transformers.convert_graph_to_onnx import convert

# Initialize wandb
wandb.init(project="azure_ml_test")

# Define the callback for logging predictions during evaluation
class WandbPredictionLoggingCallback(WandbCallback):
    def __init__(self, trainer, tokenizer, val_dataset, label_map, num_samples=100, freq=1):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset.select(range(num_samples))
        self.label_map = label_map
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if state.epoch % self.freq == 0:
            predictions, labels, _ = self.trainer.predict(self.val_dataset)
            decoded_predictions = decode_predictions(predictions, labels, self.val_dataset, self.label_map)
            predictions_df = pd.DataFrame(decoded_predictions)
            predictions_df["epoch"] = state.epoch
            predictions_table = wandb.Table(dataframe=predictions_df)
            wandb.log({"predictions": predictions_table})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "roberta-base"

# Load a finance-related dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocess by tokenizing the sentences by padding and truncating with a maximum length of 128
def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding="max_length", max_length=128)

# Preprocess the dataset
full_encoded_dataset = dataset.map(preprocess_function, batched=True)['train']

# Split the dataset
# Splits into train and then the join of test and valid
train_testvalid = full_encoded_dataset.train_test_split(test_size=0.1, seed=42)
# Splits the join of test and valid into test and valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.2, seed=42)

# Create a combined dataset for training, testing, and validation
encoded_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

# Load a model for sequence classification with the number of labels corresponding to sentiment analysis
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

# Load the metric
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="wandb",
)
# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["valid"],
    compute_metrics=compute_metrics,
)

def decode_predictions(predictions, labels, encoded_dataset, label_map):
    pred_labels = np.argmax(predictions, axis=1)
    pred_label_str = [label_map[label] for label in pred_labels]
    true_label_str = [label_map[label] for label in labels]
    sentences = encoded_dataset['sentence']
    return {
        'Sentence': sentences,
        'Prediction': pred_label_str,
        'Ground Truth': true_label_str
    }

# Instantiate the WandbPredictionLoggingCallback
prediction_logging_callback = WandbPredictionLoggingCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=encoded_dataset["valid"],
    label_map={0: 'negative', 1: 'neutral', 2: 'positive'},
    num_samples=10,
    freq=1,
)

# Add the callback to the trainer
trainer.add_callback(prediction_logging_callback)

# Train the model
trainer.train()
# Perform the prediction
predictions, labels, _ = trainer.predict(encoded_dataset["test"])

# Decode the predictions
decoded_predictions = decode_predictions(predictions, labels, encoded_dataset["test"], {0: 'negative', 1: 'neutral', 2: 'positive'})

# Create the pandas DataFrame
df = pd.DataFrame(decoded_predictions)

wandb.log({"final_predictions": wandb.Table(dataframe=df)})

# Define the base path using pathlib for better path management
base_path = Path(__file__).parent

# Save the model and tokenizer in PyTorch format
hf_model_dir = base_path / "hf_model"
hf_model_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(hf_model_dir)
tokenizer.save_pretrained(hf_model_dir)
wandb.log_model(str(hf_model_dir), name="hf_model")

# Alternative PyTorch format
torch_model_dir = base_path / "torch_model"
torch_model_dir.mkdir(parents=True, exist_ok=True)
torch_model_path = torch_model_dir / "model.pt"
torch.save(model.state_dict(), torch_model_path)
tokenizer.save_pretrained(torch_model_dir)
wandb.log_model(str(torch_model_dir), name="pt_model")

# Export to ONNX
onnx_model_dir = base_path / "onnx_model"
onnx_model_dir.mkdir(parents=True, exist_ok=True)
onnx_model_path = onnx_model_dir / "model.onnx"
convert(framework="pt", model=str(hf_model_dir), output=onnx_model_path, opset=11)
wandb.log_model(str(onnx_model_dir), name="onnx_model")

# For TensorFlow format, we need to convert the PyTorch model to TensorFlow
tensorflow_model_dir = base_path / "tf_model"
tensorflow_model_dir.mkdir(parents=True, exist_ok=True)
tf_model = TFAutoModelForSequenceClassification.from_pretrained(str(hf_model_dir), from_pt=True)
tf_model.save_pretrained(tensorflow_model_dir, saved_model=True)
tokenizer.save_pretrained(tensorflow_model_dir)
wandb.log_model(str(tensorflow_model_dir), name="tf_model")

wandb.finish()