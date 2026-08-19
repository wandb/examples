# /// script
# dependencies = ["accelerate", "fastprogress", "timm", "torcheval", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Simple_accelerate_integration_wandb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{simple-accelerate} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using Huggingface Accelerate with Weights and Biases
    <!--- @wandbcode{simple-accelerate} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Accelerate](https://github.com/huggingface/accelerate) is this amazing little framework that simplifies your PyTorch training scripts enabling you to train with all the tricks out there!
    - Quickly convert your code to support multiple hardward (GPUS, TPUs, Metal,...)
    - One code to support mixed precision, bfloat16 and even 8 bit Adam.

    Minimal code and no boilerplate. Weights and Biases integration out of the box!

    ```diff
      import torch
      import torch.nn.functional as F
      from datasets import load_dataset
    + from accelerate import Accelerator

    + accelerator = Accelerator(log_with="wandb")
    + accelerator.init_trackers("my_wandb_project", config=cfg)
    - device = 'cpu'
    + device = accelerator.device

      model = torch.nn.Transformer().to(device)
      optimizer = torch.optim.Adam(model.parameters())

      dataset = load_dataset('my_dataset')
      data = torch.utils.data.DataLoader(dataset, shuffle=True)

    + model, optimizer, data = accelerator.prepare(model, optimizer, data)

      model.train()
      for epoch in range(10):
          for source, targets in data:
              source = source.to(device)
              targets = targets.to(device)

              optimizer.zero_grad()

              output = model(source)
              loss = F.cross_entropy(output, targets)

    -         loss.backward()
    +         accelerator.backward(loss)

              optimizer.step()
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training and Image Classifier
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: accelerate wandb torcheval timm fastprogress !pip install accelerate wandb torcheval timm fastprogress
    return


@app.cell
def _():
    import os
    from types import SimpleNamespace

    import wandb

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torchvision.datasets import FashionMNIST
    import torchvision.transforms as T
    from torcheval.metrics.toolkit import sync_and_compute
    from fastprogress import progress_bar

    from accelerate import Accelerator

    return (
        Accelerator,
        AdamW,
        DataLoader,
        F,
        FashionMNIST,
        SimpleNamespace,
        T,
        nn,
        progress_bar,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Store your configuration parameters
    """)
    return


@app.cell
def _(SimpleNamespace):
    cfg = SimpleNamespace(
        path=".",
        bs=256,
        epochs=5,
        size=28,
        num_workers=8,
    )

    WANDB_PROJECT = "accelerate_fmnist"
    return WANDB_PROJECT, cfg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    setup transforms
    """)
    return


@app.cell
def _(T, cfg):
    tfms = T.Compose([
        T.RandomCrop(cfg.size, padding=1),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    return (tfms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create a simple CNN
    """)
    return


@app.cell
def _(nn):
    def conv_block(in_ch, out_ch, ks=3): return nn.Sequential(nn.BatchNorm2d(in_ch),
                                                        nn.Conv2d(in_ch, out_ch, ks, stride=2, padding=0), 
                                                        nn.ReLU())

    def create_cnn():
        return nn.Sequential(nn.Conv2d(1, 16, 5, stride=1, padding="same"),
                             conv_block(16, 32),
                             conv_block(32, 64),
                             conv_block(64, 128),
                             conv_block(128, 256, 1),
                             nn.Sequential(nn.Flatten(), nn.Linear(256,10), nn.BatchNorm1d(10)),
                            )

    return (create_cnn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Wrap everything into a training functions (this is necessary to run on multiple GPUS, if it is only one, you can skip the wrapping)
    """)
    return


@app.cell
def _(
    Accelerator,
    AdamW,
    DataLoader,
    F,
    FashionMNIST,
    WANDB_PROJECT,
    create_cnn,
    progress_bar,
    tfms,
):
    def train(cfg):

        # data
        ds = FashionMNIST(cfg.path, transform=tfms, download=True) 
        dl = DataLoader(ds, batch_size=cfg.bs, num_workers=cfg.num_workers)
    
        # model
        model = create_cnn()
    
        # training setup
        optimizer = AdamW(model.parameters(), lr=1e-3)
    
    
        # accelerate
        accelerator = Accelerator(log_with="wandb")
    
        # this will call wandb.init(...)
        accelerator.init_trackers(WANDB_PROJECT, config=cfg)
    
        # prepare
        model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    
        # train
        model.train()
        for epoch in progress_bar(range(cfg.epochs)):
            accurate, num_elems = 0., 0
            for source, targets in dl:
                optimizer.zero_grad()
                output = model(source)
                loss = F.cross_entropy(output, targets)
                accelerator.backward(loss)
            
                # under the hood this calls wandb.log(...) on the main process
                accelerator.log({"train_loss": loss})
            
                accurate_preds = output.argmax(dim=1) == targets
                num_elems += accurate_preds.shape[0]
                accurate += accurate_preds.long().sum()
                optimizer.step()
            accuracy = accurate.item() / num_elems
            accelerator.log({"epoch":epoch, "accuracy":accuracy}, log_kwargs={"wandb": {"commit": False}})
            print(f"epoch: {epoch:3} || loss: {loss:5.3f} || accuracy: {accuracy:5.3f}")
    
        # this will call wandb.finish()
        accelerator.end_training()

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's train on 2 GPUs! This is really nice, as accelerate will take care of only calling `log` on the main process, so only one run get's created, so no need to manually check the rank of the process when using multiple GPUs.
    """)
    return


@app.cell
def _(cfg, train):
    num_GPUSs = 2

    from accelerate import notebook_launcher

    notebook_launcher(train, (cfg,), num_processes=num_GPUSs)
    return


if __name__ == "__main__":
    app.run()
