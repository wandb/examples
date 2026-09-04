# /// script
# dependencies = ["lightning", "torch-tb-profiler", "torchvision", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Profile_PyTorch_Code.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{trace-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{trace-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Profiling PyTorch Code
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook demonstrates how to incorporate [PyTorch Kineto](https://github.com/pytorch/kineto)'s
    [Tensorboard plugin](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md)
    for profiling PyTorch code
    with [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)
    as the high-level training API
    and
    [Weights & Biases](https://wandb.ai/site)
    as the logging solution.

    The final result looks something like what you see below:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](https://i.imgur.com/fwSc5Z9.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The work done by processes, threads, and streams on the CPU and GPU
    is displayed along with precise timing information
    in an interactive viewer that can be incorporated into
    Weights & Biases
    [workspaces](https://docs.wandb.ai/ref/app/pages/workspaces)
    and [Reports](https://docs.wandb.ai/guides/reports)
    or exported to external viewers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That means you can incorporate tracing and profiling into
    your model training and evaluation pipeline --
    storing, visualizing, and communicating
    performance results alongside other key metrics and metadata,
    like [loss curves](https://docs.wandb.ai/guides/track/log),
    [hard examples from datasets](https://docs.wandb.ai/guides/data-vis),
    and [hyperparameter optimization results](https://docs.wandb.ai/guides/sweeps).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > _NB:_ This tool is based on the
    [Chrome Trace Viewer](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool),
    which works best with that browser.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb lightning torch_tb_profiler torchvision !pip install -q wandb lightning torch_tb_profiler torchvision
    return


@app.cell
def _():
    import glob

    import lightning.pytorch as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision
    from torchvision import datasets, transforms

    from torch.profiler import tensorboard_trace_handler
    import wandb

    # drop slow mirror from list of MNIST mirrors
    torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                          if not mirror.startswith("http://yann.lecun.com")]
    return (
        F,
        datasets,
        glob,
        nn,
        optim,
        pl,
        tensorboard_trace_handler,
        torch,
        transforms,
        wandb,
    )


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Set Up Profiled Training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Network Module
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To profile neural network code,
    we first need to write it.

    For this demo,
    we'll stick with a simple
    [LeNet](http://yann.lecun.com/exdb/lenet/)-style DNN,
    based on the
    [PyTorch introductory tutorial](https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html).
    """)
    return


@app.cell
def _(F, nn, optim, pl, torch):
    OPTIMIZERS = {
        "Adadelta": optim.Adadelta,
        "Adagrad" : optim.Adagrad,
        "SGD": optim.SGD,
    }

    class Net(pl.LightningModule):
      """Very simple LeNet-style DNN, plus DropOut."""

      def __init__(self, optimizer="Adadelta"):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.optimizer = self.set_optimizer(optimizer)

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
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

      def set_optimizer(self, optimizer):
        return OPTIMIZERS[optimizer]

    return (Net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To get this module to work with PyTorch Lightning,
    we need to define two more methods,
    which hook into the training loop.

    Check out [this tutorial video and notebook](http://wandb.me/lit-video)
    for more on using PyTorch Lightning and W&B.
    """)
    return


@app.cell
def _(F, Net):
    def training_step(self, batch, idx):
      inputs, labels = batch
      outputs = self(inputs)
      loss =  F.nll_loss(outputs, labels)

      return {"loss": loss}
    
    def configure_optimizers(self):
      return self.optimizer(self.parameters(), lr=0.1)

    Net.training_step = training_step
    Net.configure_optimizers = configure_optimizers
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Profiler Callback
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The profiler operates a bit like a PyTorch optimizer:
    it has a `.step` method that we need to call
    to demarcate the code we're interested in profiling.

    A single training step (forward and backward prop)
    is both the typical target of performance optimizations
    and already rich enough to more than fill out a profiling trace,
    so we want to call `.step` on each step.

    The cell below defines a quick-and-dirty
    method for doing so in PyTorch Lightning using the
    [`Callback` system](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html).
    """)
    return


@app.cell
def _(pl):
    class TorchTensorboardProfilerCallback(pl.Callback):
      """Quick-and-dirty Callback for invoking TensorboardProfiler during training.
  
      For greater robustness, extend the pl.profiler.profilers.BaseProfiler. See
      https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html"""

      def __init__(self, profiler):
        super().__init__()
        self.profiler = profiler 

      def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.profiler.step()
        pl_module.log_dict(outputs)  # also logging the loss, while we're here

    return (TorchTensorboardProfilerCallback,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run Profiled Training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We're now ready to go!

    The cell below creates a `DataLoader`
    based on the information in the `config`uration dictionary.
    Choices made here have a substantial impact on performance
    and show up very markedly in the trace.

    After you've run with the default values,
    check out the creation of the `trainloader`
    and the `trainer`
    for comments on what these arguments do
    and then try a few different choices out, as suggested below.
    """)
    return


@app.cell
def _(
    Net,
    TorchTensorboardProfilerCallback,
    datasets,
    glob,
    pl,
    tensorboard_trace_handler,
    torch,
    transforms,
    wandb,
):
    # initial values are defaults, for all except batch_size, which has no default
    config = {"batch_size": 32,  # try log-spaced values from 1 to 50,000
              "num_workers": 0,  # try 0, 1, and 2
              "pin_memory": False,  # try False and True
              "precision": 32,  # try 16 and 32
              "optimizer": "Adadelta",  # try optim.Adadelta and optim.SGD
              }

    with wandb.init(project="trace", config=config) as run:

        # Set up MNIST data
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        dataset = datasets.MNIST("../data", train=True, download=True,
                                transform=transform)

        ## Using a raw DataLoader, rather than LightningDataModule, for greater transparency
        trainloader = torch.utils.data.DataLoader(
          dataset,
          # Key performance-relevant configuration parameters:
          ## batch_size: how many datapoints are passed through the network at once?
          batch_size=wandb.config.batch_size,
          # larger batch sizes are more compute efficient, up to memory constraints

          ##  num_workers: how many side processes to launch for dataloading (should be >0)
          num_workers=wandb.config.num_workers,
          # needs to be tuned given model/batch size/compute

          ## pin_memory: should a fixed "pinned" memory block be allocated on the CPU?
          pin_memory=wandb.config.pin_memory,
          # should nearly always be True for GPU models, see https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
          )
    
        # Set up model
        model = Net(optimizer=wandb.config["optimizer"])

        # Set up profiler
        wait, warmup, active, repeat = 1, 1, 2, 1
        total_steps = (wait + warmup + active) * (1 + repeat)
        schedule =  torch.profiler.schedule(
          wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler = torch.profiler.profile(
          schedule=schedule, on_trace_ready=tensorboard_trace_handler("wandb/latest-run/tbprofile"), with_stack=False)

        with profiler:
            profiler_callback = TorchTensorboardProfilerCallback(profiler)

            trainer = pl.Trainer(max_epochs=1, max_steps=total_steps,
                                logger=pl.loggers.WandbLogger(log_model=True, save_code=True),
                                callbacks=[profiler_callback], precision=wandb.config.precision)

            trainer.fit(model, trainloader)

        profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
        profile_art.add_file(glob.glob("wandb/latest-run/tbprofile/*.pt.trace.json")[0], "trace.pt.trace.json")
        run.log_artifact(profile_art)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reading Profiling Results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Head to the Artifacts tab
    (identified by the
    ["stacked pucks"](https://stackoverflow.com/questions/2822650/why-is-a-database-always-represented-with-a-cylinder)
    database icon)
    for your W&B [run page](https://docs.wandb.ai/ref/app/pages/run-page),
    at the URL that appears in the output of the cell above,
    then select the artifact named `trace-`.
    In the Files tab, select `trace.pt.trace.json`
    to pull up the Trace Viewer.

    > You can also check out an example from an earlier run
    [here](https://wandb.ai/wandb/trace/artifacts/profile/trace-224bfvza/56c5d50902233baa7710/files/trace.pt.trace.json).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The trace shows which operations were running and when
    in each process/thread/stream
    on the CPU and on the GPU.

    In the main thread (the one in which the Profiler Steps appear),
    locate the following steps:
    1. the loading of data (hint: look for `enumerate` on the CPU, nothing on the GPU)
    2. the forward pass to calculate the loss (hint: look for simultaneous activity on CPU+GPU,
    with [`aten`](https://pytorch.org/cppdocs/#aten) in the operation names)
    3. the backward pass to calculate the gradient of the loss (hint: look for simultaneous activity on CPU+GPU, with [`backward`](https://pytorch.org/cppdocs/#autograd) in the operation names).

    If you ran with the default settings
    (in particular, `num_workers=0`),
    you'll notice that these steps are all run sequentially,
    meaning that between loading one batch
    and loading the next,
    the `DataLoader` is effectively idling,
    and during the loading of a batch, the GPU is idling.

    Change `num_workers` in the config to `1` or `2`
    and then re-execute the cell above.
    You should notice a difference,
    in particular in the fraction of time the GPU is active.
    (Note: the `DataLoader` may even be hard to find in this case!)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For more on how to read these results, check out
    [this W&B Report](http://wandb.me/trace-report).
    """)
    return


if __name__ == "__main__":
    app.run()
