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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Model_Management_Guide.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # W&B Model Management Guide Companion Notebook

    This is a companion notebook to the [W&B Model Management Guide](https://docs.wandb.ai/guides/models).

    **Table of Contents**
    * **Cell 1**: Installs `wandb` python library
    * **Cell 2** (Form): Allows you to specify some parameters and defines a handful of helper functions. Note: there is not any `wandb` specific library calls in these helper functions - they are purely used to allow the example cells to be more terse and focus on the key aspects of Model Management
    * **Cell 3**: (Train, Log, & Link Models) Covers steps [2. Traing & log a Model](https://docs.wandb.ai/guides/models#2.-train-and-log-model-versions) and [3. Link Model Versions to the Collection](https://docs.wandb.ai/guides/models#3.-link-model-versions-to-the-portfolio)
    * **Cell 4**: (Use, Evaluate, and Promote a Model) Covers steps [4. Using a Model Version](https://docs.wandb.ai/guides/models#4.-use-a-model-version), [5. Evaluate Model Performance](https://docs.wandb.ai/guides/models#5.-evaluate-model-performance), and [6. Promote a Version to Production](https://docs.wandb.ai/guides/models#6.-promote-a-version-to-production)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    **Stop! 🛑** Please complete [Step 1 of the tutorial](https://docs.wandb.ai/guides/models/walkthrough#1-create-a-new-registered-model) before continuing. This will ensure you have a **Model Collection** defined in your project. Enter the Project name where you created the Collection in `project_name` and the name of the Collection in the `model_collection_name` fields respectively.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -U -qqq
    return


@app.cell
def _():
    project_name = 'quickstart-model-registry'  #@param {type:"string"}
    # dataset_name = "mnist" #@param {type:"string"}
    dataset_name = 'mnist'
    model_collection_name = 'MNIST Grayscale 28x28'  #@param {type:"string"}
    import wandb
    import torch
    from torchvision import datasets, transforms
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import OneCycleLR

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            output = self.fc2(x)
            return output

    def _sample_mnist(split0, split1, is_train=True):
        """Sample MNIST dataset"""
        mnist_data = datasets.MNIST('train_data/' if is_train else 'test_data/', download=True, train=is_train, transform=transforms.Compose([transforms.ToTensor()]))
        extra = len(mnist_data) - split0 - split1
        assert extra >= 0
        splits = torch.utils.data.random_split(mnist_data, [split0, split1, extra])
        return (splits[0], splits[1])

    def build_train_data(train_size, val_size, batch_size=128):
        splits = _sample_mnist(train_size, val_size)
        return (DataLoader(splits[0], batch_size=batch_size), DataLoader(splits[1], batch_size=val_size))

    def build_test_data(test_size):
        splits = _sample_mnist(test_size, 0, is_train=False)
        return DataLoader(splits[0], batch_size=test_size)

    def build_model(learning_rate, total_steps):
        device = torch.device('cpu')
        _model = Net().to(device)
        optimizer = optim.Adam(_model.parameters())
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)
        return (_model, optimizer, scheduler)

    def train_step(model, optimizer, scheduler, batch_x, batch_y):
        _model.train()
        batch_x, batch_y = (batch_x.to('cpu'), batch_y.to('cpu'))
        optimizer.zero_grad()
        preds = _model(batch_x)
        loss = F.cross_entropy(preds, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        return (loss.item(), preds)

    @torch.no_grad()
    def evaluate_model(model, eval_dl):
        device = torch.device('cpu')
        _model.eval()
        test_loss = 0
        correct = 0
        preds = []
        for data, target in eval_dl:
            data, target = (data.to(device), target.to(device))
            output = _model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            preds += list(pred.flatten().tolist())
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(eval_dl.dataset)
        accuracy = 100.0 * correct / len(eval_dl.dataset)
        return (test_loss, accuracy, preds)  # sum up batch loss  # get the index of the max log-probability

    return (
        Net,
        build_model,
        build_test_data,
        build_train_data,
        dataset_name,
        evaluate_model,
        model_collection_name,
        project_name,
        torch,
        train_step,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train, Log, & Link Models
    """)
    return


@app.cell
def _(evaluate_model, torch, train_step, wandb):
    def save_model(model, is_best=False):
        """Save model to W&B and locally if it's the best one so far."""
        _art = wandb.Artifact(f'mnist-{wandb.run.id}', 'model')  ##### W&B MODEL MANAGEMENT SPECIFIC CALLS ######
        torch.save(_model.state_dict(), 'model.pt')
        _art.add_file('model.pt')
        wandb.log_artifact(_art, aliases=['best', 'latest'] if is_best else None)
        return _art

    def train_model(model, optimizer, scheduler, train_loader, val_loader, num_epochs=5):
        """A simple training loop"""
        best_val_loss = 10000000000.0
        best_model_art = None
        for epoch in range(num_epochs):
            for batch_x, batch_y in train_loader:
                train_loss, _ = train_step(_model, optimizer, scheduler, batch_x, batch_y)
                wandb.log({'epoch': epoch, 'train_loss': train_loss, 'learning_rate': optimizer.param_groups[0]['lr']})
            val_loss, val_acc, _ = evaluate_model(_model, val_loader)
            wandb.log({'val_loss': val_loss, 'val_acc': val_acc})
            best_val_loss = min(best_val_loss, val_loss)
            model_art = save_model(_model, is_best=val_loss <= best_val_loss)
            if val_loss <= best_val_loss:
                best_model_art = model_art
                print('New best model saved!')
            print(f'Epoch {epoch}: val_loss: {val_loss}, val_acc: {val_acc}')
        return best_model_art

    return (train_model,)


@app.cell
def _(
    build_model,
    build_train_data,
    dataset_name,
    model_collection_name,
    project_name,
    train_model,
    wandb,
):
    # Startup a W&B Run
    wandb.init(project=project_name, job_type='model_trainer', config={'train_size': 2000, 'val_size': 200, 'batch_size': 64, 'learning_rate': 0.001, 'epochs': 5})
    _config = wandb.config
    train_dl, val_dl = build_train_data(_config.train_size, _config.val_size, _config.batch_size)
    _art = wandb.Artifact(f'{dataset_name}-train', 'dataset')
    _art.add_dir('./train_data')
    wandb.use_artifact(_art)
    _model, optimizer, scheduler = build_model(_config.learning_rate, total_steps=len(train_dl) * _config.epochs)
    best_model_art = train_model(_model, optimizer, scheduler, train_dl, val_dl, _config.epochs)
    wandb.run.link_artifact(best_model_art, model_collection_name, ['latest'])
    # Load in the training data
    # (Optional) Declare dataset dependency
    # Define a model
    # Train the Model
    ##### W&B MODEL MANAGEMENT SPECIFIC CALLS ######
    # Finish the Run
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Use, Evaluate, and Promote a Model
    """)
    return


@app.cell
def _(
    Net,
    build_test_data,
    dataset_name,
    evaluate_model,
    model_collection_name,
    project_name,
    torch,
    wandb,
):
    wandb.init(project=project_name, job_type='model_evaluator', config={'test_size': 100})
    _config = wandb.config
    test_dl = build_test_data(test_size=_config.test_size)
    _art = wandb.Artifact(f'{dataset_name}-test', 'dataset')
    _art.add_dir('./test_data')
    wandb.use_artifact(_art)
    model_art = wandb.use_artifact(f'{model_collection_name}:latest')
    model_path = model_art.get_path('model.pt').download()
    _model = Net().cpu()
    checkpt = torch.load(model_path)
    _model.load_state_dict(checkpt)
    val_loss, val_acc, preds = evaluate_model(_model, test_dl)
    table = wandb.Table(data=[], columns=[])
    table.add_column('image', [wandb.Image(i.numpy()) for i in list(test_dl)[0][0]])
    table.add_column('label', list(test_dl)[0][1].tolist())
    table.add_column('pred', preds)
    wandb.log({'test_loss': val_loss, 'test_acc': val_acc, 'predictions': table})
    wandb.run.link_artifact(model_art, model_collection_name, ['latest', 'production'])
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
