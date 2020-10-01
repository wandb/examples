<img src="https://i.imgur.com/gb6B4ig.png" width="400" alt="Weights & Biases" />

<div><img /></div>

<img src="https://i.imgur.com/uEtWSEb.png" width="650" alt="Weights & Biases" />

# 🚀 Getting Started

### Never lose your progress again. 
Save everything you need to compare and reproduce models — architecture, hyperparameters, weights, model predictions, GPU usage, git commits, and even datasets — in 5 minutes. W&B is free for personal use and academic projects, and it's easy to get started.

[**→ Check out our library of example scripts →**](https://github.com/wandb/examples/tree/master/examples)
or read on for code snippets and more!

# 🤝 Simple integration with any framework
Install wandb library and login:
```
pip install wandb
wandb login
```
Flexible integration for any Python script:
```python
import wandb

# 1. Start a W&B run
wandb.init(project='gpt3')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01

# Model training code here ...

# 3. Log metrics over time to visualize performance
for i in range (10):
    wandb.log({"loss": loss})
```

### [Try in a colab →](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb)

If you have any questions, please don't hesitate to ask in our [Slack community](http://bit.ly/slack-forum).

![](https://i.imgur.com/TU34QFZ.png)

**[Explore a W&B dashboard](https://www.youtube.com/watch?v=gnD8BFuyVUA)**

# 📈 Track model and data pipeline hyperparameters
Set `wandb.config` once at the beginning of your script to save your hyperparameters, input settings (like dataset name or model type), and any other independent variables for your experiments. This is useful for analyzing your experiments and reproducing your work in the future. Setting configs also allows you to [visualize](https://docs.wandb.com/sweeps/visualize-sweep-results) the relationships between features of your model architecture or data pipeline and the model performance (as seen in the screenshot above).

```python
wandb.init()
wandb.config.epochs = 4
wandb.config.batch_size = 32
wandb.config.learning_rate = 0.001
wandb.config.architecture = "resnet"
```

- **[See how to set configs in a colab →](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-config/Configs_in_W%26B.ipynb)**
- [Docs](https://docs.wandb.com/library/config)

# 🏗 Frameworks

## 📍 Keras
Use the Keras callback to automatically save all the metrics and the loss values tracked in `model.fit`. To get you started here's a minimal example.
```python
# Import W&B
import wandb
from wandb.keras import WandbCallback

# Step1: Initialize W&B run
wandb.init(project='project_name')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01

# Model training code here ...

# Step 3: Add WandbCallback 
model.fit(X_train, y_train,  validation_data=(X_test, y_test),
          callbacks=[WandbCallback()])
```

- **[Try in a colab →](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Simple_Keras_Integration.ipynb)**
- [Learn More](https://app.wandb.ai/wandb/getting-started/reports/Keras--VmlldzoyMTEwNjQ)
- [Docs](https://docs.wandb.com/frameworks/keras)

## 📍 PyTorch
W&B provides first class support for PyTorch. To automatically log gradients and store the network topology, you can call `watch` and pass in your PyTorch model.
```python
import wandb

# 1. Start a new run
wandb.init(project="gpt-3")

# 2. Save model inputs and hyperparameters
config = wandb.config
config.dropout = 0.01

# 3. Log gradients and model parameters
wandb.watch(model)
for batch_idx, (data, target) in enumerate(train_loader):
  ...  
  if batch_idx % args.log_interval == 0:      
    # 4. Log metrics to visualize performance
    wandb.log({"loss": loss})
```

- **[Try in a colab →](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)**
- [Learn More](https://app.wandb.ai/wandb/getting-started/reports/Pytorch--VmlldzoyMTEwNzM)
- [Docs](https://docs.wandb.com/frameworks/pytorch)


## 📍 TensorFlow
The simplest way to log metrics in TensorFlow is by logging `tf.summary` with the TensorFlow logger.
```python
import wandb

# 1. Start a W&B run
wandb.init(project='gpt3')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01

# Model training here

# 3. Log metrics over time to visualize performance
with tf.Session() as sess:
  # ...
  wandb.tensorflow.log(tf.summary.merge_all())
```

- **[Try in a colab →](https://colab.research.google.com/drive/126c1k5IfbQpE7dVmhnoDTdmFfC7CgJqg?usp=sharing)**
- [Docs](https://docs.wandb.com/frameworks/tensorflow)


## 📍 Scikit
You can use wandb to visualize and compare your scikit-learn models' performance with just a few lines of code.
```python
import wandb
wandb.init(project="visualize-sklearn")

# Model training here

# Log classifier visualizations
wandb.sklearn.plot_classifier(clf, X_train, X_test, y_train, y_test, y_pred, y_probas, labels, model_name='SVC', feature_names=None)

# Log regression visualizations
wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test,  model_name='Ridge')

# Log clustering visualizations
wandb.sklearn.plot_clusterer(kmeans, X_train, cluster_labels, labels=None, model_name='KMeans')
```

- **[Try in a colab →](https://colab.research.google.com/drive/1dxWV5uulLOQvMoBBaJy2dZ3ZONr4Mqlo?usp=sharing)**
- [Docs](https://docs.wandb.com/library/integrations/scikit)


## 📍 fastai
Visualize, compare, and iterate on fastai models using Weights & Biases with the WandbCallback.
```python
import wandb
fastai2.callback.wandb import WandbCallback

# 1. Start a new run
wandb.init(project="gpt-3")

# 2. Automatically log model metrics
learn.fit(..., cbs=WandbCallback())
```

- **[Try in a colab →](http://bit.ly/fastai-wandb)**
- [Docs](https://docs.wandb.com/library/integrations/fastai)


## 📍 HuggingFace
Run a script with the Trainer, which automatically logs losses, evaluation metrics, model topology and gradients
```python
# 1. Install the wandb library
pip install wandb

# 2. Run a script that has the Trainer to automatically logs metrics, model topology and gradients
python run_glue.py \
 --model_name_or_path bert-base-uncased \
 --task_name MRPC \
 --data_dir $GLUE_DIR/$TASK_NAME \
 --do_train \
 --evaluate_during_training \
 --max_seq_length 128 \
 --per_gpu_train_batch_size 32 \
 --learning_rate 2e-5 \
 --num_train_epochs 3 \
 --output_dir /tmp/$TASK_NAME/ \
 --overwrite_output_dir \
 --logging_steps 50
```

- **[Try in a colab →](https://colab.research.google.com/drive/1NEiqNPhiouu2pPwDAVeFoN4-vTYMz9F8?usp=sharing)**
- [Docs](https://docs.wandb.com/library/integrations/huggingface)

# 🧹 Optimize hyperparameters with Sweeps
Use Weights & Biases Sweeps to automate hyperparameter optimization and explore the space of possible models.

### [Get started in 5 mins →](https://docs.wandb.com/sweeps/quickstart)

### Benefits of using W&B Sweeps 
- **Quick setup:** With just a few lines of code you can run W&B sweeps.
- **Transparent:** We cite all the algorithms we're using, and our code is open source.
- **Powerful:** Our sweeps are completely customizable and configurable. You can launch a sweep across dozens of machines, and it's just as easy as starting a sweep on your laptop.

<img src="https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-LyfPCyvV8By5YBltxfh%2F-LyfQsxswLC-6WKGgfGj%2Fcentral%20sweep%20server%203.png?alt=media&token=c81e4fe7-7ee4-48ea-a4cd-7b28113c6088" width="400" alt="Weights & Biases" />

### Common use cases
- **Explore:** Efficiently sample the space of hyperparameter combinations to discover promising regions and build an intuition about your model.
- **Optimize:**  Use sweeps to find a set of hyperparameters with optimal performance.
- **K-fold cross validation:** Here's a brief code example of k-fold cross validation with W&B Sweeps.

### Visualize Sweeps results
The hyperparameter importance plot surfaces which hyperparameters were the best predictors of, and highly correlated to desirable values for your metrics.

<img src="https://paper-attachments.dropbox.com/s_194708415DEC35F74A7691FF6810D3B14703D1EFE1672ED29000BA98171242A5_1578695757573_image.png" width="720" alt="Weights & Biases" />

Parallel coordinates plots map hyperparameter values to model metrics. They're useful for honing in on combinations of hyperparameters that led to the best model performance.

<img src="https://i.imgur.com/THYXBN0.png" width="720" alt="Weights & Biases" />

# 📜 Reports
Reports let you [organize visualizations, describe your findings, and share updates with collaborators](https://www.youtube.com/watch?v=o2dOSIDDr1w&&ab_channel=Weights%26Biases).

### Common use cases
- **Notes:** Add a graph with a quick note to yourself.
- **Collaboration:** Share findings with your colleagues.
- **Work log:** Track what you've tried, and plan next steps.

**Explore reports in [The Gallery →](https://app.wandb.ai/gallery) | [Docs](https://docs.wandb.com/reports)**

Once you have experiments in W&B, you can visualize and document results in Reports with just a few clicks. Here's a quick [demo video](https://www.youtube.com/watch?v=jWBGKGAjt6w&t=2s&ab_channel=Weights%26Biases).

![](https://i.imgur.com/dn0Dyd8.png)

If you have any questions, please don't hesitate to ask in our [Slack community](http://bit.ly/slack-forum).
