# /// script
# dependencies = ["einops", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-model-registry/New_Model_Logging_in_W&B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Logging and Registering Models in W&B
    It's never been easier to log your model checkpoints, keep track of the best ones, and maintain lineage of runs and results!

    W&B is introducing a few convenience methods to make logging models and linking them to the registry simple:
    - `log_model`
    - `use_model`
    - `link_model`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb einops !pip install -qqq wandb einops
    return


@app.cell
def _():
    import torch
    from torch import nn
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange
    from torch.utils.data import Dataset
    from torchvision import transforms
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import wandb

    return (
        Adam,
        DataLoader,
        Dataset,
        Rearrange,
        nn,
        rearrange,
        repeat,
        torch,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Log in to W&B
    - You can explicitly login using `wandb login` or `wandb.login()` (See below)
    - Alternatively you can set environment variables. There are several env variables which you can set to change the behavior of W&B logging. The most important are:
        - `WANDB_API_KEY` - create a new API key in your "Settings" section under your profile at [wandb.ai/settings](https://wandb.ai/settings)
        - `WANDB_BASE_URL` - this is the url of the W&B server
    - Create a new API key in "Profile" -> "Settings" in the W&B App. Store your API key securely. It can only be viewed once when created.

    ![api_token](https://drive.google.com/uc?export=view&id=1Xn7hnn0rfPu_EW0A_-32oCXqDmpA0-kx)
    """)
    return


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Define the Model and Dataset
    This is a simple implementation of a Vision Transformer (ViT) and utilizes a random dataset for training.
    - Credit to https://github.com/lucidrains/vit-pytorch
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define some config for the model and dataset
    """)
    return


@app.cell
def _():
    # Define the number of samples, classes, and image size
    num_samples = 100
    num_classes = 10
    image_size = 256
    batch_size = 32
    return batch_size, image_size, num_classes, num_samples


@app.cell
def _(
    DataLoader,
    Dataset,
    Rearrange,
    batch_size,
    image_size,
    nn,
    num_classes,
    num_samples,
    rearrange,
    repeat,
    torch,
):
    # helpers
    def pair(t):
        return t if isinstance(t, tuple) else (t, t)

    # classes
    class FeedForward(nn.Module):
        def __init__(self, dim, hidden_dim, dropout = 0.):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            return self.net(x)

    class Attention(nn.Module):
        def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
            super().__init__()
            inner_dim = dim_head *  heads
            project_out = not (heads == 1 and dim_head == dim)

            self.heads = heads
            self.scale = dim_head ** -0.5

            self.norm = nn.LayerNorm(dim)

            self.attend = nn.Softmax(dim = -1)
            self.dropout = nn.Dropout(dropout)

            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()

        def forward(self, x):
            x = self.norm(x)

            qkv = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)

    class Transformer(nn.Module):
        def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.layers = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(dim, mlp_dim, dropout = dropout)
                ]))

        def forward(self, x):
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x

            return self.norm(x)

    class ViT(nn.Module):
        def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
            super().__init__()
            image_height, image_width = pair(image_size)
            patch_height, patch_width = pair(patch_size)

            assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

            num_patches = (image_height // patch_height) * (image_width // patch_width)
            patch_dim = channels * patch_height * patch_width
            assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )

            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.dropout = nn.Dropout(emb_dropout)

            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

            self.pool = pool
            self.to_latent = nn.Identity()

            self.mlp_head = nn.Linear(dim, num_classes)

        def forward(self, img):
            x = self.to_patch_embedding(img)
            b, n, _ = x.shape

            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)

            x = self.transformer(x)

            x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

            x = self.to_latent(x)
            return self.mlp_head(x)


    # Define a custom dataset
    class RandomImageDataset(Dataset):
        def __init__(self, num_samples, num_classes, image_size):
            self.num_samples = num_samples
            self.num_classes = num_classes
            self.image_size = image_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate a random image tensor
            image = torch.randn(3, self.image_size, self.image_size)  # 3 channels, image_size x image_size
            # Generate a random label
            label = torch.randint(0, self.num_classes, (1,)).item()
            return image, label



    # Create the dataset
    dataset = RandomImageDataset(num_samples=num_samples, num_classes=num_classes, image_size=image_size)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return ViT, dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log Model Checkpoints to W&B with 1 Line!

    Use the `log_model` method to log a model artifact containing the contents inside the ‘path’ to an a run. It also marks it as an output to the run. You can see the full lineage graph of the model artifact by accessing the [lineage](https://docs.wandb.ai/guides/artifacts/explore-and-traverse-an-artifact-graph#docusaurus_skipToContent_fallback) tab inside the Artifacts view.

    `log_model()` takes as input:

    - `path`: A path to the model file(s), which can be a local file (of the form `/local/directory/file.txt`), directory (of the form `/local/directory`), or reference path to S3 (`s3://bucket/path`).
    - `name`: An optional name for the model artifact the files will be logged to. Note that if no name is specified, This will default to the basename of the input path prepended with the run ID.
    - `aliases`: An optional list of aliases, which can be thought of as semantic ‘nicknames’ or identifiers for a model version. For example, if this model yielded the best accuracy, you might add the alias ‘highest-accuracy’ or ‘best’.
    """)
    return


@app.cell
def _(Adam, ViT, dataloader, image_size, nn, num_classes, torch, wandb):
    run = wandb.init(project="new_model_logging",
                     job_type="training")

    v = ViT(
        image_size = image_size,
        patch_size = 32,
        num_classes = num_classes,
        dim = 128,
        depth = 3,
        heads = 2,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(v.parameters(), lr=0.003)

    # Training loop
    best_accuracy = 0
    for epoch in range(5):  # number of epochs
        for images, labels in dataloader:
            # Forward pass
            preds = v(images)
            loss = criterion(preds, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train/loss": loss})

        # Model evaluation after each epoch (using a validation set)
        # Here you would write your validation loop and calculate accuracy
        val_accuracy = 0.5  # Assume this is the validation accuracy you compute
        model_path = 'model_vit.pth'
        torch.save(v.state_dict(), model_path)

        # Check if this is the best model so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # Log the model to your W&B run
            wandb.log_model(name=f"model_vit-{wandb.run.id}", path=model_path, aliases=["best", f"epoch_{epoch}"])
        else:
            wandb.log_model(name=f"model_vit-{wandb.run.id}", path=model_path, aliases=[f"epoch_{epoch}"])
    return (run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Link Your Best Models to the Model Registry
    You can bookmark your best model checkpoints and centralize them across your team. The Model Registry allows you can organize your best models by task, manage model lifecycle, facilitate easy tracking and auditing throughout the ML lifecyle, and automate downstream actions with webhooks or jobs. You can this via api through `link_model()`, which takes as input:

    - `path`: A path to the model file(s), which can be a local file (of the form `/local/directory/file.txt`), directory (of the form `/local/directory`), or reference path to S3 (`s3://bucket/path`).
    - `registered_model_name`: the name of the Registered Model - a collection of linked model versions in the Model Registry, typically representing a team’s ML task - that the model should be linked to. If no Registered Model with the given name exists, a new one will be created with this name.
    - `name`: An **optional** name for the model artifact the files will be logged to. Note that if no name is specified, This will default to the basename of the input path prepended with the run ID.
    - `aliases`: An **optional** list of aliases, which can be thought of as semantic ‘nicknames’ or identifiers for a linked model version. For example, since this model is being linked, or published, to the Model Registry, you might add an alias “staging” or “QA”.
    """)
    return


@app.cell
def _(run, wandb):
    # Link the best model to the W&B Model Registry (after all epochs are finished)
    artifact_name = f"model_vit-{wandb.run.id}:best"
    best_model_path = wandb.use_model(artifact_name)

    # Link the best model to the registry
    wandb.link_model(path=best_model_path,
                     registered_model_name="Industrial ViT",
                     aliases=["staging"])
    run.finish()
    return


if __name__ == "__main__":
    app.run()
