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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Track_PyTorch_Lightning_with_Fabric_and_Wandb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{fabric_colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{fabric_colab} -->

    # ⚡ Track PyTorch Lightning with Fabric and Wandb
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At Weights & Biases, we love anything
    that makes training deep learning models easier.
    That's why we worked with the folks at PyTorch Lightning to
    [integrate our experiment tracking tool](https://docs.wandb.com/library/integrations/lightning)
    directly into the Fabric library of PyTorch Lightning

    [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) is a lightweight wrapper for organizing your PyTorch code and easily adding advanced features such as distributed training and 16-bit precision.
    It retains all the flexibility of PyTorch,
    in case you need it,
    but adds some useful abstractions
    and builds in some best practices.

    [Pytorch Fabric](https://lightning.ai/docs/fabric/stable/) allows you to scale PyTorch models on
    distributed machines while
    maintaining full control of your
    training loop.

    ## What this notebook covers:

    1. How to get basic metric logging with the `WandbLogger`
    2. How to log media with W&B

    ## The interactive dashboard in W&B will look like this:
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !pip install wandb
    return


@app.cell
def _():
    import os
    os.environ["WANDB_API_KEY"]=""
    return (os,)


@app.cell
def _():
    import wandb
    wandb.login()
    return (wandb,)


@app.cell
def _():
    import lightning as L
    import torch; import torchvision as tv
    from wandb.integration.lightning.fabric import WandbLogger

    return L, WandbLogger, torch, tv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 💡 Tracking Experiments with WandbLogger

    PyTorch Lightning has a `WandbLogger` to easily log your experiments with Wights & Biases. Just pass it to your `Trainer` to log to W&B. See the WandbLogger docs for all parameters. Note, to log the metrics to a specific W&B Team, pass your Team name to the `entity` argument in `WandbLogger`

    #### `lightning.fabric.loggers.WandbLogger()`

    | Functionality | Argument/Function | PS |
    | ------ | ------ | ------ |
    | Logging models | `WandbLogger(... ,log_model='all')` or `WandbLogger(... ,log_model=True`) | Log all models if `log_model="all"` and at end of training if `log_model=True`
    | Set custom run names | `WandbLogger(... ,name='my_run_name'`) | |
    | Organize runs by project | `WandbLogger(... ,project='my_project')` | |
    | Log histograms of gradients and parameters | `WandbLogger.watch(model)`  | `WandbLogger.watch(model, log='all')` to log parameter histograms  |
    | Log hyperparameters | Call `self.save_hyperparameters()` within `LightningModule.__init__()` |
    | Log custom objects (images, audio, video, molecules…) | Use `WandbLogger.log_text`, `WandbLogger.log_image` and `WandbLogger.log_table`, etc. |
    """)
    return


@app.cell
def _(WandbLogger):
    logger = WandbLogger(project="Cifar10_ptl_fabric")
    return (logger,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Log custom hyperparameters and configurations
    """)
    return


@app.cell
def _(logger):
    lr = 0.001
    batch_size = 16
    num_epochs = 5
    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    log_images_after_n_batches = 200

    logger.log_hyperparams({
        "lr": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "classes": classes,
        "log_images_after_n_batches": log_images_after_n_batches
    })
    return batch_size, classes, log_images_after_n_batches, lr, num_epochs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save Data to Weights and Biases Artifacts

    This allows us to audit and create direct data lineages to our experiments
    """)
    return


@app.cell
def _():
    root_folder = "data"
    return (root_folder,)


@app.cell
def _(root_folder, tv):
    train_dataset = tv.datasets.CIFAR10(root_folder, download=True,
                                  train=True,
                                  transform=tv.transforms.ToTensor())
    test_dataset = tv.datasets.CIFAR10(root_folder, download=True,
                                  train=False,
                                  transform=tv.transforms.ToTensor())
    return test_dataset, train_dataset


@app.cell
def _(train_dataset):
    data_folder = train_dataset.base_folder # same as test_dataset.base_folder
    return (data_folder,)


@app.cell
def _(data_folder, logger, os, root_folder, wandb):
    data_art = wandb.Artifact(name="cifar10", type="dataset")
    data_art.add_dir(os.path.join(root_folder, data_folder))
    logger.experiment.log_artifact(data_art)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configure our Model and Training
    """)
    return


@app.cell
def _(lr, torch, tv):
    model = tv.models.resnet18()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, optimizer


@app.cell
def _(wandb):
    class TableLoggingCallback:
        def __init__(self, wandb_logger):
            self.wandb_logger = wandb_logger
            self.table = wandb.Table(columns=["image", "prediction", "ground_truth"])

        def on_test_batch_end(self, images, predictions, ground_truths):
            for image, prediction, ground_truth in zip(images, predictions, ground_truths):
                self.table.add_data(wandb.Image(image), prediction, ground_truth)

        def on_model_epoch_end(self):
            prediction_table = self.table
            print(self.table.data[0])
            self.wandb_logger.experiment.log({"prediction_table": prediction_table}) # You can directly access the run object via `experiment`

            # We could also use
            # (1) wandb_logger.log_metrics()
            # (2) wandb_logger.log_table() <Note: this method would mean changing how the callback constructs tables>

            self.table = wandb.Table(columns=["image", "prediction", "ground_truth"])

    return (TableLoggingCallback,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load our model, datasources, and loggers into PyTorch Fabric
    """)
    return


@app.cell
def _(TableLoggingCallback, logger):
    tlc = TableLoggingCallback(logger)
    return (tlc,)


@app.cell
def _(L, logger, tlc):
    fabric = L.Fabric(loggers=[logger], callbacks=[tlc])
    fabric.launch()
    return (fabric,)


@app.cell
def _(
    batch_size,
    fabric,
    model,
    optimizer,
    test_dataset,
    torch,
    train_dataset,
):
    model_1, optimizer_1 = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size))
    test_dataloader = fabric.setup_dataloaders(torch.utils.data.DataLoader(test_dataset, batch_size=batch_size))
    return model_1, optimizer_1, test_dataloader, train_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run training and log test predictions

    For every epoch, run a training step and a test step. For each n test batches, we log the batch of test images caption by the prediction and label, and we create a wandb.Table() in which to store test predictions using our custom callback
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No additional dependencies outside the Torch modeling you're used to!
    """)
    return


@app.cell
def _(logger, model_1):
    logger.watch(model_1)
    return


@app.cell
def _(
    classes,
    fabric,
    log_images_after_n_batches,
    logger,
    model_1,
    num_epochs,
    optimizer_1,
    test_dataloader,
    torch,
    train_dataloader,
):
    model_1.train()
    for epoch in range(num_epochs):
        fabric.print(f'Epoch: {epoch}')
        cum_loss = 0  # Training Loop
        for batch in train_dataloader:
            inputs, labels = batch
            optimizer_1.zero_grad()
            outputs = model_1(inputs)  # Batch by batch of data from training dataset
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            cum_loss = cum_loss + loss.item()
            fabric.backward(loss)
            optimizer_1.step()
            fabric.log_dict({'loss': loss.item()})
        fabric.log_dict({'avg_loss': cum_loss / len(train_dataloader)})
        correct = 0
        total = 0
        class_correct = list((0.0 for i in range(10)))
        class_total = list((0.0 for i in range(10)))  # Stream per batch training metrics
        test_batch_ctr = 0
        with torch.no_grad():  # Stream per epoch training metrics
            for batch_ctr, batch in enumerate(test_dataloader):
                images, labels = batch  # Validation Loop
                outputs = model_1(images)
                _, predicted = torch.max(outputs, 1)
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                for i in range(batch[0].size(0)):
                    label = labels[i]
                    class_correct[label] = class_correct[label] + c[i].item()
                    class_total[label] = class_total[label] + 1
                if batch_ctr % log_images_after_n_batches == 0:  # Batch by batch of data from testing dataset
                    predictions = [classes[prediction] for prediction in predicted]
                    label_names = [classes[truth] for truth in labels]
                    loggable_images = [image for image in images]
                    captions = [f'pred: {pred}\nlabel: {truth}' for pred, truth in zip(predictions, label_names)]
                    logger.log_image(key='test_image_batch', images=loggable_images, step=None, caption=captions)
                    fabric.call('on_test_batch_end', images=loggable_images, predictions=predictions, ground_truths=label_names)  # Overall Test Accuracy
        test_acc = 100 * correct / total
        class_acc = {f'{classes[i]}_acc': 100 * class_correct[i] / class_total[i] for i in range(10) if class_total[i] > 0}
        loggable_dict = {'test_acc': test_acc}
        loggable_dict.update(class_acc)  # Per Class Accuracy
        fabric.log_dict(loggable_dict)
        fabric.call('on_model_epoch_end')  # Test Images labeled with Class prediction for qualitative analysis  # Automatically construct and log wandb.Images  # Can also just directly log the below list via fabric.log_dict  # [wandb.Image(image, caption=classes[predicted]) for image, predicted, label in zip(images, predicted, labels)])  # Populate per batch data within our table  # Calculate cumulative test metrics  # Stream per epoch validation metrics  # Save epoch test data table to dashboard
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finish our experiment!
    """)
    return


@app.cell
def _(logger):
    logger.experiment.finish()
    return


if __name__ == "__main__":
    app.run()
