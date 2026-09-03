# /// script
# dependencies = ["wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{tables_quickstart} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="300" alt="Weights & Biases" />

    <!--- @wandbcode{tables_quickstart} -->

    # View & analyze model predictions during training

    This quickstart guide covers how to track, visualize, and compare model predictions over the course of training, using PyTorch on MNIST data.

    With [W&B Tables](https://docs.wandb.com/datasets-and-predictions):
    1. Log metrics, images, text, etc. to a `wandb.Table()` during model training or evaluation
    2. View, sort, filter, group, join, interactively query, and explore these tables
    3. Compare model predictions or results: dynamically across specific images, hyperparameters/model versions, or time steps.

    # Examples
    ## Compare predicted scores for specific images

    [Live example: compare predictions after 1 vs 5 epochs of training →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)
    <img src="https://i.imgur.com/NMme6Qj.png" alt="1 epoch vs 5 epochs of training"/>
    The histograms compare per-class scores between the two models. The top green bar in each histogram represents model "CNN-2, 1 epoch" (id 0), which only trained for 1 epoch. The bottom purple bar represents model "CNN-2, 5 epochs" (id 1), which trained for 5 epochs. The images are filtered to cases where the models disagree. For example, in the first row, the "4" gets high scores across all the possible digits after 1 epoch, but after 5 epochs it scores highest on the correct label and very low on the rest.

    ## Focus on top errors over time
    [Live example →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

    See incorrect predictions (filter to rows where "guess" != "truth") on the full test data. Note that there are 229 wrong guesses after 1 training epoch, but only 98 after 5 epochs.
    <img src="https://i.imgur.com/7g8nodn.png" alt="side by side, 1 vs 5 epochs of training"/>

    ## Compare model performance and find patterns

    [See full detail in a live example →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

    Filter out correct answers, then group by the guess to see examples of misclassified images and the underlying distribution of true labels—for two models side-by-side. A model variant with 2X the layer sizes and learning rate is on the left, and the baseline is on the right. Note that the baseline makes slightly more mistakes for each guessed class.
    <img src="https://i.imgur.com/i5PP9AE.png" alt="grouped errors for baseline vs double variant"/>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sign up or login

    [Sign up or login](https://wandb.ai/login) to W&B to see and interact with your experiments in the browser.

    In this example we're using Google Colab as a convenient hosted environment, but you can run your own training scripts from anywhere and visualize metrics with W&B's experiment tracking tool.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -qqq
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    log to your account
    """)
    return


@app.cell
def _():
    import wandb

    WANDB_PROJECT = "mnist-viz"
    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 0. Setup

    Install dependencies, download MNIST, and create train and test datasets using PyTorch.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T 
    import torch.nn.functional as F


    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # create train and test dataloaders
    def get_dataloader(is_train, batch_size, slice=5):
        "Get a training dataloader"
        ds = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
        loader = torch.utils.data.DataLoader(dataset=ds, 
                                             batch_size=batch_size, 
                                             shuffle=True if is_train else False, 
                                             pin_memory=True, num_workers=2)
        return loader

    return F, get_dataloader, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Define the model and training schedule

    * Set the number of epochs to run, where each epoch consists of a training step and a validation (test) step. Optionally configure the amount of data to log per test step. Here the number of batches and number of images per batch to visualize are set low to simplify the demo.
    * Define a simple convolutional neural net (following [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) code).
    * Load in train and test sets using PyTorch
    """)
    return


@app.cell
def _(get_dataloader, nn, torch):
    # Number of epochs to run
    # Each epoch includes a training step and a test step, so this sets
    # the number of tables of test predictions to log
    EPOCHS = 1
    NUM_BATCHES_TO_LOG = 10
    # Number of batches to log from the test data for each test step
    # (default set low to simplify demo)
    NUM_IMAGES_PER_BATCH = 32  #79
    NUM_CLASSES = 10
    # Number of images to log per test batch
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001  #128
    L1_SIZE = 32
    # training configuration and hyperparameters
    L2_SIZE = 64
    CONV_KERNEL_SIZE = 5

    class ConvNet(nn.Module):

    # changing this may require changing the shape of adjacent layers
        def __init__(self, num_classes=10):
            super(ConvNet, self).__init__()
    # define a two-layer convolutional neural network
            self.layer1 = nn.Sequential(nn.Conv2d(1, L1_SIZE, CONV_KERNEL_SIZE, stride=1, padding=2), nn.BatchNorm2d(L1_SIZE), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(nn.Conv2d(L1_SIZE, L2_SIZE, CONV_KERNEL_SIZE, stride=1, padding=2), nn.BatchNorm2d(L2_SIZE), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(7 * 7 * L2_SIZE, NUM_CLASSES)
            self.softmax = nn.Softmax(NUM_CLASSES)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out
    train_loader = get_dataloader(is_train=True, batch_size=BATCH_SIZE)
    test_loader = get_dataloader(is_train=False, batch_size=2 * BATCH_SIZE)
    device_1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # uncomment to see the shape of a given layer:  #print("x: ", x.size())
    return (
        BATCH_SIZE,
        CONV_KERNEL_SIZE,
        ConvNet,
        EPOCHS,
        L1_SIZE,
        L2_SIZE,
        LEARNING_RATE,
        NUM_BATCHES_TO_LOG,
        NUM_CLASSES,
        NUM_IMAGES_PER_BATCH,
        device_1,
        test_loader,
        train_loader,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Run training and log test predictions

    For every epoch, run a training step and a test step. For each test step, create a wandb.Table() in which to store test predictions. These can be visualized, dynamically queried, and compared side by side in your browser.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    CONV_KERNEL_SIZE,
    ConvNet,
    EPOCHS,
    F,
    L1_SIZE,
    L2_SIZE,
    LEARNING_RATE,
    NUM_BATCHES_TO_LOG,
    NUM_CLASSES,
    NUM_IMAGES_PER_BATCH,
    device_1,
    nn,
    test_loader,
    torch,
    train_loader,
    wandb,
):
    # ✨ W&B: Initialize a new run to track this model's training
    wandb.init(project='table-quickstart')
    cfg = wandb.config
    # ✨ W&B: Log hyperparameters using config
    cfg.update({'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE, 'l1_size': L1_SIZE, 'l2_size': L2_SIZE, 'conv_kernel': CONV_KERNEL_SIZE, 'img_count': min(10000, NUM_IMAGES_PER_BATCH * NUM_BATCHES_TO_LOG)})
    model = ConvNet(NUM_CLASSES).to(device_1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
    # define model, loss, and optimizer
        scores = F.softmax(outputs.data, dim=1)
        log_scores = scores.cpu().numpy()
        log_images = images.cpu().numpy()
        log_labels = labels.cpu().numpy()
    # convenience funtion to log predictions for a batch of test images
        log_preds = predicted.cpu().numpy()
        _id = 0  # obtain confidence scores for all classes
        for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
            img_id = str(_id) + '_' + str(log_counter)
            test_table.add_data(img_id, wandb.Image(i), p, l, *s)
            _id = _id + 1
            if _id == NUM_IMAGES_PER_BATCH:
                break  # adding ids based on the order of the images
    total_step = len(train_loader)
    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(train_loader):  # add required info to data table:
            images = images.to(device_1)  # id, image pixels, model's guess, true label, scores for all classes
            labels = labels.to(device_1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # train the model
            wandb.log({'loss': loss})
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, EPOCHS, i + 1, total_step, loss.item()))  # training step
        columns = ['id', 'image', 'guess', 'truth']
        for digit in range(10):
            columns.append('score_' + str(digit))
        test_table = wandb.Table(columns=columns)  # forward pass
        model.eval()
        log_counter = 0
        with torch.no_grad():  # backward and optimize
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device_1)
                labels = labels.to(device_1)  # ✨ W&B: Log loss over training steps, visualized in the UI live
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                if log_counter < NUM_BATCHES_TO_LOG:
                    log_test_predictions(images, labels, outputs, predicted, test_table, log_counter)
                    log_counter = log_counter + 1
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum().item()  # ✨ W&B: Create a Table to store predictions for each test step
            acc = 100 * correct / total
            wandb.log({'epoch': epoch, 'acc': acc})
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc))
        wandb.log({'test_predictions': test_table})
    # ✨ W&B: Mark the run as complete (useful for multi-cell notebook)
    wandb.finish()  # test the model  # ✨ W&B: Log accuracy across training epochs, to visualize in the UI  # ✨ W&B: Log predictions table to wandb
    return


if __name__ == "__main__":
    app.run()
