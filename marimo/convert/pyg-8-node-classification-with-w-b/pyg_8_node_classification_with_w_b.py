# /// script
# dependencies = ["https://data-pyg-org/whl/torch-${torch}-html", "pytorch_geometric @ git+https://github.com/pyg-team/pytorch_geometric.git", "torch-scatter", "torch-sparse", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pyg/8_Node_Classification_(with_W&B).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{pytorch_geometric_example} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{pytorch_geometric_example} -->
    """)
    return


@app.cell
def _():
    # Install required packages.

    # packages added via marimo's package management: torch-scatter https://data.pyg.org/whl/torch-${TORCH}.html !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
    # packages added via marimo's package management: torch-sparse https://data.pyg.org/whl/torch-${TORCH}.html !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
    # packages added via marimo's package management: git+https://github.com/pyg-team/pytorch_geometric.git !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
    # packages added via marimo's package management: wandb !pip install -qqq wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Setup and login to Weights & Biases
    """)
    return


@app.cell
def _():
    enable_wandb = True
    if enable_wandb:
        import wandb
    return enable_wandb, wandb


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell
def _():
    import os
    import pdb
    import torch
    import pandas

    os.environ['TORCH'] = torch.__version__
    print(torch.__version__)
    return pandas, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Helper function for visualization.
    """)
    return


@app.cell
def _(pandas, wandb):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    def visualize(h, color):
        z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
        plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
        plt.show()

    def embedding_to_wandb(h, color, key="embedding"):
        num_components = h.shape[-1]
        df = pandas.DataFrame(data=h.detach().cpu().numpy(),
                            columns=[f"c_{i}" for i in range(num_components)])
        df["target"] = color.detach().cpu().numpy().astype("str")
        cols = df.columns.tolist()
        df = df[cols[-1:] + cols[:-1]]
        wandb.log({key: df})

    return embedding_to_wandb, visualize


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Node Classification with Graph Neural Networks

    [Previous: Introduction: Hands-on Graph Neural Networks](https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8)

    This tutorial will teach you how to apply **Graph Neural Networks (GNNs) to the task of node classification**.
    Here, we are given the ground-truth labels of only a small subset of nodes, and want to infer the labels for all the remaining nodes (*transductive learning*).

    To demonstrate, we make use of the `Cora` dataset, which is a **citation network** where nodes represent documents.
    Each node is described by a 1433-dimensional bag-of-words feature vector.
    Two documents are connected if there exists a citation link between them.
    The task is to infer the category of each document (7 in total).

    This dataset was first introduced by [Yang et al. (2016)](https://arxiv.org/abs/1603.08861) as one of the datasets of the `Planetoid` benchmark suite.
    We again can make use [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) for an easy access to this dataset via [`torch_geometric.datasets.Planetoid`](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid):
    """)
    return


@app.cell
def _():
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures



    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    return data, dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Overall, this dataset is quite similar to the previously used [`KarateClub`](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.KarateClub) network.
    We can see that the `Cora` network holds 2,708 nodes and 10,556 edges, resulting in an average node degree of 3.9.
    For training this dataset, we are given the ground-truth categories of 140 nodes (20 for each class).
    This results in a training node label rate of only 5%.

    In contrast to `KarateClub`, this graph holds the additional attributes `val_mask` and `test_mask`, which denotes which nodes should be used for validation and testing.
    Furthermore, we make use of **[data transformations](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-transforms) via `transform=NormalizeFeatures()`**.
    Transforms can be used to modify your input data before inputting them into a neural network, *e.g.*, for normalization or data augmentation.
    Here, we [row-normalize](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.NormalizeFeatures) the bag-of-words input feature vectors.

    We can further see that this network is undirected, and that there exists no isolated nodes (each document has at least one citation).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training a Multi-layer Perception Network (MLP)

    In theory, we should be able to infer the category of a document solely based on its content, *i.e.* its bag-of-words feature representation, without taking any relational information into account.

    Let's verify that by constructing a simple MLP that solely operates on input node features (using shared weights across all nodes):
    """)
    return


@app.cell
def _(dataset, torch):
    from torch.nn import Linear
    import torch.nn.functional as F

    class MLP(torch.nn.Module):

        def __init__(self, hidden_channels):
            super().__init__()
            torch.manual_seed(12345)
            self.lin1 = Linear(dataset.num_features, hidden_channels)
            self.lin2 = Linear(hidden_channels, dataset.num_classes)

        def forward(self, x):
            x = self.lin1(x)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin2(x)
            return x
    model = MLP(hidden_channels=16)
    print(model)
    return F, MLP


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    (optionally) logging the data attributes to W&B summary.
    """)
    return


@app.cell
def _(data, dataset, enable_wandb, wandb):
    if enable_wandb:
        wandb.init(project='node-classification')
        summary = dict()
        summary["data"] = dict()
        summary["data"]["num_features"] = dataset.num_features
        summary["data"]["num_classes"] = dataset.num_classes
        summary["data"]["num_nodes"] = data.num_nodes
        summary["data"]["num_edges"] = data.num_edges 
        summary["data"]["has_isolated_nodes"] = data.has_isolated_nodes()
        summary["data"]["has_self_nodes"] = data.has_self_loops()
        summary["data"]["is_undirected"] = data.is_undirected()
        summary["data"]["num_training_nodes"] = data.train_mask.sum()
        wandb.summary = summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our MLP is defined by two linear layers and enhanced by [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html?highlight=relu#torch.nn.ReLU) non-linearity and [dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html?highlight=dropout#torch.nn.Dropout).
    Here, we first reduce the 1433-dimensional feature vector to a low-dimensional embedding (`hidden_channels=16`), while the second linear layer acts as a classifier that should map each low-dimensional node embedding to one of the 7 classes.

    Let's train our simple MLP by following a similar procedure as described in [the first part of this tutorial](https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8).
    We again make use of the **cross entropy loss** and **Adam optimizer**.
    This time, we also define a **`test` function** to evaluate how well our final model performs on the test node set (which labels have not been observed during training).

    We also visualize the embeddings of the untrained model to in visually comparing the progress made by the training process below.

    **NOTE**: *For W&B mode, please set up the embedding projector from the setting panel of the logged table. More information can be found here: https://docs.wandb.ai/ref/app/features/panels/weave/embedding-projector*
    """)
    return


@app.cell
def _(
    MLP,
    data,
    display,
    embedding_to_wandb,
    enable_wandb,
    torch,
    visualize,
    wandb,
):
    from IPython.display import Javascript
    display(Javascript('google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'))
    model_1 = MLP(hidden_channels=16)
    with torch.no_grad():
        _out = model_1(data.x)
    if enable_wandb:
        embedding_to_wandb(_out, color=data.y, key='mlp/embedding/init')
    else:
        visualize(_out, data.y)
    _criterion = torch.nn.CrossEntropyLoss()
    _optimizer = torch.optim.Adam(model_1.parameters(), lr=0.01, weight_decay=0.0005)

    def _train():
        model_1.train()
        _optimizer.zero_grad()
        _out = model_1(data.x)
        _loss = _criterion(_out[data.train_mask], data.y[data.train_mask])
        _loss.backward()
        _optimizer.step()
        return _loss

    def test():
        model_1.eval()
        _out = model_1(data.x)
        pred = _out.argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        return test_acc
    for _epoch in range(1, 201):
        _loss = _train()
        if enable_wandb:
            wandb.log({'mlp/loss': _loss})
        print(f'Epoch: {_epoch:03d}, Loss: {_loss:.4f}')
    return Javascript, model_1, test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After training the model, we can call the `test` function to see how well our model performs on unseen labels.
    Here, we are interested in the accuracy of the model, *i.e.*, the ratio of correctly classified nodes:

    We also visualize the embeddings of the output. This will give us a visual hint as to how good the model is performing, when compared to the embeddings of the geometric models defined below.
    """)
    return


@app.cell
def _(data, embedding_to_wandb, enable_wandb, model_1, test, visualize, wandb):
    test_acc = test()
    _out = model_1(data.x)
    if enable_wandb:
        embedding_to_wandb(_out, color=data.y, key='mlp/embedding/trained')
        wandb.summary['mlp/accuracy'] = test_acc
        wandb.log({'mlp/accuracy': test_acc})
    else:
        visualize(_out, data.y)
    print(f'Test Accuracy: {test_acc:.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As one can see, our MLP performs rather bad with only about 59% test accuracy.
    But why does the MLP do not perform better?
    The main reason for that is that this model suffers from heavy overfitting due to only having access to a **small amount of training nodes**, and therefore generalizes poorly to unseen node representations.

    It also fails to incorporate an important bias into the model: **Cited papers are very likely related to the category of a document**.
    That is exactly where Graph Neural Networks come into play and can help to boost the performance of our model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training a Graph Neural Network (GNN)

    We can easily convert our MLP to a GNN by swapping the `torch.nn.Linear` layers with PyG's GNN operators.

    Following-up on [the first part of this tutorial](https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8), we replace the linear layers by the [`GCNConv`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv) module.
    To recap, the **GCN layer** ([Kipf et al. (2017)](https://arxiv.org/abs/1609.02907)) is defined as

    $$
    \mathbf{x}_v^{(\ell + 1)} = \mathbf{W}^{(\ell + 1)} \sum_{w \in \mathcal{N}(v) \, \cup \, \{ v \}} \frac{1}{c_{w,v}} \cdot \mathbf{x}_w^{(\ell)}
    $$

    where $\mathbf{W}^{(\ell + 1)}$ denotes a trainable weight matrix of shape `[num_output_features, num_input_features]` and $c_{w,v}$ refers to a fixed normalization coefficient for each edge.
    In contrast, a single `Linear` layer is defined as

    $$
    \mathbf{x}_v^{(\ell + 1)} = \mathbf{W}^{(\ell + 1)} \mathbf{x}_v^{(\ell)}
    $$

    which does not make use of neighboring node information.
    """)
    return


@app.cell
def _(F, dataset, torch):
    from torch_geometric.nn import GCNConv

    class GCN(torch.nn.Module):

        def __init__(self, hidden_channels):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = GCNConv(dataset.num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x
    model_2 = GCN(hidden_channels=16)
    print(model_2)
    return (GCN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's visualize the node embeddings of our **untrained** GCN network.
    For visualization, we make use of [**TSNE**](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) to embed our 7-dimensional node embeddings onto a 2D plane.
    """)
    return


@app.cell
def _(GCN, data, embedding_to_wandb, enable_wandb, visualize):
    model_3 = GCN(hidden_channels=16)
    model_3.eval()
    _out = model_3(data.x, data.edge_index)
    if enable_wandb:
        embedding_to_wandb(_out, color=data.y, key='gcn/embedding/init')
    else:
        visualize(_out, data.y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We certainly can do better by training our model.
    The training and testing procedure is once again the same, but this time we make use of the node features `x` **and** the graph connectivity `edge_index` as input to our GCN model.
    """)
    return


@app.cell
def _(GCN, Javascript, data, display, enable_wandb, torch, wandb):
    display(Javascript('google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'))
    model_4 = GCN(hidden_channels=16)
    if enable_wandb:
        wandb.watch(model_4)
    _optimizer = torch.optim.Adam(model_4.parameters(), lr=0.01, weight_decay=0.0005)
    _criterion = torch.nn.CrossEntropyLoss()

    def _train():
        model_4.train()
        _optimizer.zero_grad()
        _out = model_4(data.x, data.edge_index)
        _loss = _criterion(_out[data.train_mask], data.y[data.train_mask])
        _loss.backward()
        _optimizer.step()
        return _loss

    def test_1():
        model_4.eval()
        _out = model_4(data.x, data.edge_index)
        pred = _out.argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        return test_acc
    for _epoch in range(1, 101):
        _loss = _train()
        if enable_wandb:
            wandb.log({'gcn/loss': _loss})
        print(f'Epoch: {_epoch:03d}, Loss: {_loss:.4f}')
    return model_4, test_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After training the model, we can check its test accuracy:
    """)
    return


@app.cell
def _(test_1):
    test_acc_1 = test_1()
    print(f'Test Accuracy: {test_acc_1:.4f}')
    return (test_acc_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **There it is!**
    By simply swapping the linear layers with GNN layers, we can reach **81.5% of test accuracy**!
    This is in stark contrast to the 59% of test accuracy obtained by our MLP, indicating that relational information plays a crucial role in obtaining better performance.

    We can also verify that once again by looking at the output embeddings of our **trained** model, which now produces a far better clustering of nodes of the same category.
    """)
    return


@app.cell
def _(
    data,
    embedding_to_wandb,
    enable_wandb,
    model_4,
    test_acc_1,
    visualize,
    wandb,
):
    model_4.eval()
    _out = model_4(data.x, data.edge_index)
    if enable_wandb:
        wandb.summary['gcn/accuracy'] = test_acc_1
        wandb.log({'gcn/accuracy': test_acc_1})
        embedding_to_wandb(_out, color=data.y, key='gcn/embedding/trained')
        wandb.finish()
    else:
        visualize(_out, data.y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using W&B Sweeps
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this section, we'll look into how we can use [W&B Sweeps](https://wandb.ai/site/sweeps/) to perform a hyper-parameter search for the GCN. For this to work, it is essential for wandb to be enabled, i.e., `enable_wandb` should be set to `True`.
    """)
    return


@app.cell
def _(enable_wandb):
    assert enable_wandb, "W&B not enabled. Please, enable W&B and restart the notebook"
    return


@app.cell
def _(GCN, data, embedding_to_wandb, torch, wandb):
    import tqdm

    def agent_fn():
        wandb.init()
        model = GCN(hidden_channels=wandb.config.hidden_channels)
        wandb.watch(model)
        with torch.no_grad():
            _out = model(data.x, data.edge_index)
            embedding_to_wandb(_out, color=data.y, key='gcn/embedding/init')
        _optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
        _criterion = torch.nn.CrossEntropyLoss()

        def _train():
            model.train()
            _optimizer.zero_grad()
            _out = model(data.x, data.edge_index)
            _loss = _criterion(_out[data.train_mask], data.y[data.train_mask])  # Clear gradients.
            _loss.backward()  # Perform a single forward pass.
            _optimizer.step()  # Compute the loss solely based on the training nodes.
            return _loss  # Derive gradients.
      # Update parameters based on gradients.
        def test():
            model.eval()
            _out = model(data.x, data.edge_index)
            pred = _out.argmax(dim=1)
            test_correct = pred[data.test_mask] == data.y[data.test_mask]
            test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Use the class with highest probability.
            return test_acc  # Check against ground-truth labels.
        for _epoch in tqdm.tqdm(range(1, 101)):  # Derive ratio of correct predictions.
            _loss = _train()
            wandb.log({'gcn/loss': _loss})
        model.eval()
        _out = model(data.x, data.edge_index)
        test_acc = test()
        wandb.summary['gcn/accuracy'] = test_acc
        wandb.log({'gcn/accuracy': test_acc})
        embedding_to_wandb(_out, color=data.y, key='gcn/embedding/trained')
        wandb.finish()

    return (agent_fn,)


@app.cell
def _(wandb):
    sweep_config = {
        "name": "gcn-sweep",
        "method": "bayes",
        "metric": {
            "name": "gcn/accuracy",
            "goal": "maximize",
        },
        "parameters": {
            "hidden_channels": {
                "values": [8, 16, 32]
            },
            "weight_decay": {
                "distribution": "normal",
                "mu": 5e-4,
                "sigma": 1e-5,
            },
            "lr": {
                "min": 1e-4,
                "max": 1e-3
            }
        }
    }

    # Register the Sweep with W&B
    sweep_id = wandb.sweep(sweep_config, project="node-classification")
    return (sweep_id,)


@app.cell
def _(agent_fn, sweep_id, wandb):
    # Run the Sweeps agent
    wandb.agent(sweep_id, project="node-classification", function=agent_fn, count=50)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    In this chapter, you have seen how to apply GNNs to real-world problems, and, in particular, how they can effectively be used for boosting a model's performance.
    In the next section, we will look into how GNNs can be used for the task of graph classification.

    [Next: Graph Classification with Graph Neural Networks](https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (Optional) Exercises

    1. To achieve better model performance and to avoid overfitting, it is usually a good idea to select the best model based on an additional validation set.
    The `Cora` dataset provides a validation node set as `data.val_mask`, but we haven't used it yet.
    Can you modify the code to select and test the model with the highest validation performance?
    This should bring test performance to **82% accuracy**.

    2. How does `GCN` behave when increasing the hidden feature dimensionality or the number of layers?
    Does increasing the number of layers help at all?

    3. You can try to use different GNN layers to see how model performance changes. What happens if you swap out all `GCNConv` instances with [`GATConv`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv) layers that make use of attention? Try to write a 2-layer `GAT` model that makes use of 8 attention heads in the first layer and 1 attention head in the second layer, uses a `dropout` ratio of `0.6` inside and outside each `GATConv` call, and uses a `hidden_channels` dimensions of `8` per head.
    """)
    return


@app.cell
def _(F, data, torch):
    from torch_geometric.nn import GATConv

    class GAT(torch.nn.Module):

        def __init__(self, hidden_channels, heads):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = GATConv(...)
            self.conv2 = GATConv(...)

        def forward(self, x, edge_index):
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            return x
    model_5 = GAT(hidden_channels=8, heads=8)
    print(model_5)
    _optimizer = torch.optim.Adam(model_5.parameters(), lr=0.005, weight_decay=0.0005)
    _criterion = torch.nn.CrossEntropyLoss()

    def _train():
        model_5.train()
        _optimizer.zero_grad()
        _out = model_5(data.x, data.edge_index)
        _loss = _criterion(_out[data.train_mask], data.y[data.train_mask])
        _loss.backward()
        _optimizer.step()
        return _loss

    def test_2(mask):
        model_5.eval()
        _out = model_5(data.x, data.edge_index)
        pred = _out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
        return acc
    for _epoch in range(1, 201):
        _loss = _train()
        val_acc = test_2(data.val_mask)
        test_acc_2 = test_2(data.test_mask)
        print(f'Epoch: {_epoch:03d}, Loss: {_loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc_2:.4f}')
    return


if __name__ == "__main__":
    app.run()
