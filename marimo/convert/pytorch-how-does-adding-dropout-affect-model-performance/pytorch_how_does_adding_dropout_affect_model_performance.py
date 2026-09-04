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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/How_does_adding_dropout_affect_model_performance.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{pytorch-dropout} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!-- @wandbcode{pytorch-dropout} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this colab, we'll see an example of adding dropout to a PyTorch model and observe the effect dropout has on the model's performance by tracking our models in [Weights & Biases](https://wandb.ai/wandb/getting-started/reports/Visualize-Debug-Machine-Learning-Models--VmlldzoyNzY5MDk).

    You can read more about using dropout in PyTorch [here](https://wandb.ai/authors/ayusht/reports/Dropout-in-PyTorch-An-Example--VmlldzoxNTgwOTE).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -qU
    return


@app.cell
def _():
    import wandb

    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell
def _():
    import torch
    from torch import nn
    from torch import optim
    from torch.nn import functional as F
    import torchvision
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    import matplotlib.pyplot as plt
    import numpy as np

    return F, nn, np, optim, plt, torch, torchvision, transforms


@app.cell
def _(torch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download dataset and prepare `DataLoader`s
    """)
    return


@app.cell
def _(torch, torchvision, transforms):
    BATCH_SIZE = 32

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    CLASS_NAMES = ("plane", "car", "bird", "cat",
                   "deer", "dog", "frog", "horse", "ship", "truck")
    return CLASS_NAMES, testloader, trainloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualize data
    """)
    return


@app.cell
def _(CLASS_NAMES, np, plt):
    def show_batch(image_batch, label_batch):
      plt.figure(figsize=(10,10))
      for n in range(25):
          ax = plt.subplot(5,5,n+1)
          img = image_batch[n] / 2 + 0.5     # unnormalize
          img = img.numpy()
          plt.imshow(np.transpose(img, (1, 2, 0)))
          plt.title(CLASS_NAMES[label_batch[n]])
          plt.axis("off")

    return (show_batch,)


@app.cell
def _(show_batch, trainloader):
    sample_images, sample_labels = next(iter(trainloader))
    show_batch(sample_images, sample_labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Define training
    """)
    return


@app.cell
def _(torch, wandb):
    def train(model, device, train_loader, optimizer, criterion, epoch, steps_per_epoch=20):
        model.train()
        train_loss = 0
        train_total = 0
        train_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader, start=0):
            data, target = (data.to(device), target.to(device))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss = train_loss + loss.item()
            scores, predictions = torch.max(output.data, 1)
            train_total = train_total + target.size(0)
            train_correct = train_correct + int(sum(predictions == target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = round(train_correct / train_total * 100, 2)
        print('Epoch [{}], Loss: {}, Accuracy: {}'.format(_epoch, train_loss / train_total, acc), end='')
        wandb.log({'Train Loss': train_loss / train_total, 'Train Accuracy': acc, 'Epoch': _epoch})

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Define testing
    """)
    return


@app.cell
def _(torch, wandb):
    def test(model, device, test_loader, criterion, classes):
        model.eval()  # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
        test_loss = 0
        test_total = 0
        test_correct = 0
        example_images = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = (data.to(device), target.to(device))
                output = model(data)
                test_loss = test_loss + criterion(output, target).item()
                scores, predictions = torch.max(output.data, 1)  # Load the input features and labels from the test dataset
                test_total = test_total + target.size(0)
                test_correct = test_correct + int(sum(predictions == target))
        acc = round(test_correct / test_total * 100, 2)  # Make predictions: Pass image data from test dataset, make predictions about class image belongs to (0-9 in this case)
        print(' Test_loss: {}, Test_accuracy: {}'.format(test_loss / test_total, acc))
        wandb.log({'Test Loss': test_loss / test_total, 'Test Accuracy': acc})  # Compute the loss sum up batch loss

    return (test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training the unregularized model
    """)
    return


@app.cell
def _(F, nn, torch):
    class Net(nn.Module):
      def __init__(self, input_shape=(3,32,32)):
        super(Net, self).__init__()
    
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
    
        self.pool = nn.MaxPool2d(2,2)

        n_size = self._get_conv_output(input_shape)
    
        self.fc1 = nn.Linear(n_size, 512)
        self.fc2 = nn.Linear(512, 10)

      def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

      def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x
      
      def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    return (Net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Initialize Model, Loss and Optimizer
    """)
    return


@app.cell
def _(Net, device, nn, optim):
    net = Net().to(device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    return criterion, net, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Train
    """)
    return


@app.cell
def _(
    CLASS_NAMES,
    criterion,
    device,
    net,
    optimizer,
    test,
    testloader,
    train,
    trainloader,
    wandb,
):
    wandb.init(project='dropout')
    wandb.watch(net, log='all')
    for _epoch in range(8):
        train(net, device, trainloader, optimizer, criterion, _epoch)
        test(net, device, testloader, criterion, CLASS_NAMES)
    print('Finished Training')
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training a model with dropout regularization
    """)
    return


@app.cell
def _(F, Net, nn, torch):
    class Net_1(nn.Module):

        def __init__(self, input_shape=(3, 32, 32)):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3)
            self.conv2 = nn.Conv2d(32, 64, 3)
            self.conv3 = nn.Conv2d(64, 128, 3)
            self.pool = nn.MaxPool2d(2, 2)
            n_size = self._get_conv_output(input_shape)
            self.fc1 = nn.Linear(n_size, 512)
            self.fc2 = nn.Linear(512, 10)
            self.dropout = nn.Dropout(0.25)

        def _get_conv_output(self, shape):
            batch_size = 1
            input = torch.autograd.Variable(torch.rand(batch_size, *shape))
            output_feat = self._forward_features(input)
            n_size = output_feat.data.view(batch_size, -1).size(1)
            return n_size

        def _forward_features(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x

        def forward(self, x):
            x = self._forward_features(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    return (Net_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Initialize Model, Loss and Optimizer
    """)
    return


@app.cell
def _(Net_1, device, nn, optim):
    net_1 = Net_1().to(device)
    print(net_1)
    criterion_1 = nn.CrossEntropyLoss()
    optimizer_1 = optim.Adam(net_1.parameters())
    return criterion_1, net_1, optimizer_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Train
    """)
    return


@app.cell
def _(
    CLASS_NAMES,
    criterion_1,
    device,
    net_1,
    optimizer_1,
    test,
    testloader,
    train,
    trainloader,
    wandb,
):
    wandb.init(project='dropout')
    wandb.watch(net_1, log='all')
    for _epoch in range(8):
        train(net_1, device, trainloader, optimizer_1, criterion_1, _epoch)
        test(net_1, device, testloader, criterion_1, CLASS_NAMES)
    print('Finished Training')
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    # Visualize and debug model pipelines with W&B
    Think of [W&B](https://www.wandb.com/) like GitHub for machine learning models — **save everything you need to debug, compare and reproduce your models** — architecture, hyperparameters, weights, model predictions, GPU usage, git commits, and even datasets — with a few lines of code.

    W&B lightweight integrations work with any Python script, and all you need to do is sign up for a free W&B account to start tracking and visualizing your models.

    Used by the likes of OpenAI, Lyft, Github and researchers at top machine learning labs across the world, W&B is part of the new standard of best practices for machine learning.

    How W&B can help you optimize your machine learning workflows:

    - [Debug](https://wandb.ai/wandb/getting-started/reports/Visualize-Debug-Machine-Learning-Models--VmlldzoyNzY5MDk#Free-2) model performance in real time
    - Automatically tracked [GPU, CPU usage](https://wandb.ai/wandb/getting-started/reports/Visualize-Debug-Machine-Learning-Models--VmlldzoyNzY5MDk#System-4) and other system metrics
    - Powerful [custom charts](https://wandb.ai/wandb/customizable-charts/reports/Powerful-Custom-Charts-To-Debug-Model-Peformance--VmlldzoyNzY4ODI)
    - [Share model insights](https://wandb.ai/wandb/getting-started/reports/Visualize-Debug-Machine-Learning-Models--VmlldzoyNzY5MDk#Share-8) interactively
    - Efficient [hyperparameter optimization](https://docs.wandb.com/sweeps)
    - Dataset and model [pipeline tracking](https://docs.wandb.com/artifacts) and production model management

    **W&B is free for individuals, academics and open source projects.**

    ![W&B Dashboard](https://api.wandb.ai/files/wandb/images/projects/26571/5efd7117.png)
    """)
    return


if __name__ == "__main__":
    app.run()
