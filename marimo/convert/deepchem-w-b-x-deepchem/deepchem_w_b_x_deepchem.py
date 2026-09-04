# /// script
# dependencies = ["deepchem", "dgl-cu110", "dgllife", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W&B_x_DeepChem.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{deepchem} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{deepchem, v=examples} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to Graph Convolutions

    In this tutorial we will learn more about "graph convolutions." These are one of the most powerful deep learning tools for working with molecular data. The reason for this is that molecules can be naturally viewed as graphs.

    ![Molecular Graph](https://github.com/deepchem/deepchem/blob/master/examples/tutorials/basic_graphs.gif?raw=1)

    Note how standard chemical diagrams of the sort we're used to from high school lend themselves naturally to visualizing molecules as graphs. In the remainder of this tutorial, we'll dig into this relationship in significantly more detail. This will let us get a deeper understanding of how these systems work.

    ## Setup

    To run DeepChem within Colab, you'll need to run the following installation commands. This will take about 5 minutes to run to completion and install your environment. You can of course run this tutorial locally if you prefer. In that case, don't run these cells since they will download and install Anaconda on your local machine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Install Weights & Biases and log in.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: deepchem wandb !pip install -qU deepchem wandb
    return


@app.cell
def _():
    import wandb

    import warnings
    warnings.filterwarnings('ignore')
    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What are Graph Convolutions?

    Consider a standard convolutional neural network (CNN) of the sort commonly used to process images.  The input is a grid of pixels.  There is a vector of data values for each pixel, for example the red, green, and blue color channels.  The data passes through a series of convolutional layers.  Each layer combines the data from a pixel and its neighbors to produce a new data vector for the pixel.  Early layers detect small scale local patterns, while later layers detect larger, more abstract patterns.  Often the convolutional layers alternate with pooling layers that perform some operation such as max or min over local regions.

    Graph convolutions are similar, but they operate on a graph.  They begin with a data vector for each node of the graph (for example, the chemical properties of the atom that node represents).  Convolutional and pooling layers combine information from connected nodes (for example, atoms that are bonded to each other) to produce a new data vector for each node.

    # Training a GraphConvModel

    Let's use the MoleculeNet suite to load the Tox21 dataset. To featurize the data in a way that graph convolutional networks can use, we set the featurizer option to `'GraphConv'`. The MoleculeNet call returns a training set, a validation set, and a test set for us to use. It also returns `tasks`, a list of the task names, and `transformers`, a list of data transformations that were applied to preprocess the dataset. (Most deep networks are quite finicky and require a set of data transformations to ensure that training proceeds stably.)
    """)
    return


@app.cell
def _():
    import deepchem as dc
    tasks, _datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
    train_dataset, valid_dataset, test_dataset = _datasets
    return dc, tasks, test_dataset, train_dataset, transformers, valid_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also need to evaluate the performance of the model while we are training. For this, we need to define a metric, a measure of model performance. `dc.metrics` holds a collection of metrics already. For this dataset, it is standard to use the ROC-AUC score, the area under the receiver operating characteristic curve (which measures the tradeoff between precision and recall). Luckily, the ROC-AUC score is already available in DeepChem.
    """)
    return


@app.cell
def _(dc):
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    return (metric,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will import and set up WandbLogger in order to log our information to Weights & Biases. WandbLogger by default will log training loss.

    We also create a `ValidationCallback` to handle the validation scoring during training. At the interval specified, it will log the calculated metrics to Weights & Biases.
    """)
    return


@app.cell
def _(metric, transformers, valid_dataset):
    from deepchem.models.wandblogger import WandbLogger
    from deepchem.models.callbacks import ValidationCallback

    wandblogger = WandbLogger(project='deepchem_graphconv', name='basic')
    vc_valid = ValidationCallback(valid_dataset, interval=100, metrics=[metric], transformers=transformers)
    return ValidationCallback, WandbLogger, vc_valid, wandblogger


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's now train a graph convolutional network on this dataset. DeepChem has the class `GraphConvModel` that wraps a standard graph convolutional architecture underneath the hood for user convenience. Let's instantiate an object of this class and train it on our dataset.
    """)
    return


@app.cell
def _(dc, tasks, train_dataset, vc_valid, wandblogger):
    n_tasks = len(tasks)
    model = dc.models.GraphConvModel(n_tasks, mode='classification', wandb_logger=wandblogger)
    model.fit(train_dataset, nb_epoch=50, callbacks=[vc_valid])
    wandblogger.finish()
    return model, n_tasks


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To get the final performance of the model, we call `model.evaluate()`.
    """)
    return


@app.cell
def _(metric, model, test_dataset, train_dataset, transformers, valid_dataset):
    train_score = model.evaluate(train_dataset, [metric], transformers)
    valid_score = model.evaluate(valid_dataset, [metric], transformers)
    test_score = model.evaluate(test_dataset, [metric], transformers)

    print('Training set score:', train_score)
    print('Validation set score:', valid_score)
    print('Test set score:', test_score)
    return test_score, train_score, valid_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can save our results in a wandb.Table for better visualization in the dashboard. Our table will compare the training, validation, and testing ROC-AUC score for three different models on the same dataset/task: a basic GCN, a custom GCN, and a Graph Attention Network.
    """)
    return


@app.cell
def _(test_score, train_score, valid_score, wandb):
    columns = ["run_name", "train", "val", "test"]
    metrics_table = wandb.Table(columns=columns)

    # Add a row for the Basic GCN
    metrics_table.add_data("Basic GCN", train_score, valid_score, test_score)
    return (metrics_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The results are pretty good, and `GraphConvModel` is very easy to use. But what's going on under the hood? Could we build GraphConvModel ourselves? Of course! DeepChem provides Keras layers for all the calculations involved in a graph convolution. We are going to apply the following layers from DeepChem.

    -  `GraphConv` layer: This layer implements the graph convolution. The graph convolution combines per-node feature vectures in a nonlinear fashion with the feature vectors for neighboring nodes.  This "blends" information in local neighborhoods of a graph.

    - `GraphPool` layer: This layer does a max-pooling over the feature vectors of atoms in a neighborhood. You can think of this layer as analogous to a max-pooling layer for 2D convolutions but which operates on graphs instead.

    - `GraphGather`: Many graph convolutional networks manipulate feature vectors per graph-node. For a molecule for example, each node might represent an atom, and the network would manipulate atomic feature vectors that summarize the local chemistry of the atom. However, at the end of the application, we will likely want to work with a molecule level feature representation. This layer creates a graph level feature vector by combining all the node-level feature vectors.

    Apart from this we are going to apply standard neural network layers such as [Dense](https://keras.io/api/layers/core_layers/dense/), [BatchNormalization](https://keras.io/api/layers/normalization_layers/batch_normalization/) and [Softmax](https://keras.io/api/layers/activation_layers/softmax/) layer.
    """)
    return


@app.cell
def _(n_tasks):
    from deepchem.models.layers import GraphConv, GraphPool, GraphGather
    import tensorflow as tf
    import tensorflow.keras.layers as layers

    batch_size = 100

    class MyGraphConvModel(tf.keras.Model):

      def __init__(self):
        super(MyGraphConvModel, self).__init__()
        self.gc1 = GraphConv(128, activation_fn=tf.nn.tanh)
        self.batch_norm1 = layers.BatchNormalization()
        self.gp1 = GraphPool()

        self.gc2 = GraphConv(128, activation_fn=tf.nn.tanh)
        self.batch_norm2 = layers.BatchNormalization()
        self.gp2 = GraphPool()

        self.dense1 = layers.Dense(256, activation=tf.nn.tanh)
        self.batch_norm3 = layers.BatchNormalization()
        self.readout = GraphGather(batch_size=batch_size, activation_fn=tf.nn.tanh)

        self.dense2 = layers.Dense(n_tasks*2)
        self.logits = layers.Reshape((n_tasks, 2))
        self.softmax = layers.Softmax()

      def call(self, inputs):
        gc1_output = self.gc1(inputs)
        batch_norm1_output = self.batch_norm1(gc1_output)
        gp1_output = self.gp1([batch_norm1_output] + inputs[1:])

        gc2_output = self.gc2([gp1_output] + inputs[1:])
        batch_norm2_output = self.batch_norm1(gc2_output)
        gp2_output = self.gp2([batch_norm2_output] + inputs[1:])

        dense1_output = self.dense1(gp2_output)
        batch_norm3_output = self.batch_norm3(dense1_output)
        readout_output = self.readout([batch_norm3_output] + inputs[1:])

        logits_output = self.logits(self.dense2(readout_output))
        return self.softmax(logits_output)

    return MyGraphConvModel, batch_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now see more clearly what is happening.  There are two convolutional blocks, each consisting of a `GraphConv`, followed by batch normalization, followed by a `GraphPool` to do max pooling.  We finish up with a dense layer, another batch normalization, a `GraphGather` to combine the data from all the different nodes, and a final dense layer to produce the global output.

    Let's now create the DeepChem model which will be a wrapper around the Keras model that we just created. We will also specify the loss function so the model know the objective to minimize.
    """)
    return


@app.cell
def _(MyGraphConvModel, WandbLogger, dc):
    wandblogger_1 = WandbLogger(project='deepchem_graphconv', name='custom')
    model_1 = dc.models.KerasModel(MyGraphConvModel(), loss=dc.models.losses.CategoricalCrossEntropy(), wandb_logger=wandblogger_1)
    return model_1, wandblogger_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What are the inputs to this model?  A graph convolution requires a complete description of each molecule, including the list of nodes (atoms) and a description of which ones are bonded to each other.  In fact, if we inspect the dataset we see that the feature array contains Python objects of type `ConvMol`.
    """)
    return


@app.cell
def _(test_dataset):
    test_dataset.X[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Models expect arrays of numbers as their inputs, not Python objects.  We must convert the `ConvMol` objects into the particular set of arrays expected by the `GraphConv`, `GraphPool`, and `GraphGather` layers.  Fortunately, the `ConvMol` class includes the code to do this, as well as to combine all the molecules in a batch to create a single set of arrays.

    The following code creates a Python generator that given a batch of data generates the lists of inputs, labels, and weights whose values are Numpy arrays. `atom_features` holds a feature vector of length 75 for each atom. The other inputs are required to support minibatching in TensorFlow. `degree_slice` is an indexing convenience that makes it easy to locate atoms from all molecules with a given degree. `membership` determines the membership of atoms in molecules (atom `i` belongs to molecule `membership[i]`). `deg_adjs` is a list that contains adjacency lists grouped by atom degree. For more details, check out the [code](https://github.com/deepchem/deepchem/blob/master/deepchem/feat/mol_graphs.py).
    """)
    return


@app.cell
def _(batch_size, n_tasks):
    from deepchem.metrics import to_one_hot
    from deepchem.feat.mol_graphs import ConvMol
    import numpy as np

    def data_generator(dataset, epochs=1):
      for ind, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(batch_size, epochs,
                                                                       deterministic=False, pad_batches=True)):
        multiConvMol = ConvMol.agglomerate_mols(X_b)
        inputs = [multiConvMol.get_atom_features(), multiConvMol.deg_slice, np.array(multiConvMol.membership)]
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
          inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
        labels = [to_one_hot(y_b.flatten(), 2).reshape(-1, n_tasks, 2)]
        weights = [w_b]
        yield (inputs, labels, weights)

    return (data_generator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we can train the model using `fit_generator(generator)` which will use the generator we've defined to train the model.
    """)
    return


@app.cell
def _(data_generator, model_1, train_dataset, wandblogger_1):
    model_1.fit_generator(data_generator(train_dataset, epochs=50))
    wandblogger_1.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have trained our graph convolutional method, let's evaluate its performance. We again have to use our defined generator to evaluate model performance.
    """)
    return


@app.cell
def _(
    data_generator,
    metric,
    metrics_table,
    model_1,
    test_dataset,
    train_dataset,
    transformers,
    valid_dataset,
):
    train_score2 = model_1.evaluate_generator(data_generator(train_dataset), [metric], transformers)
    valid_score2 = model_1.evaluate_generator(data_generator(valid_dataset), [metric], transformers)
    test_score2 = model_1.evaluate_generator(data_generator(test_dataset), [metric], transformers)
    metrics_table.add_data('Custom GCN', train_score2, valid_score2, test_score2)
    # Add a row to the metrics table for our custom GCN
    print('Training set score:', train_score2)
    print('Validation set score:', valid_score2)
    print('Test set score:', test_score2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Success! The model we've constructed behaves nearly identically to `GraphConvModel`.

    We can also use other graph models provided by Deepchem such as the [Graph Attention Model](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#gatmodel).

    In order to use it, we must first install DGL and DGL-LifeSci as specified in the GAT documentation.

    Creating and training a GAT model follows the exact same process as the previous two models.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: dgl-cu110 !pip install --quiet dgl-cu110
    # packages added via marimo's package management: dgllife !pip install --quiet dgllife
    return


@app.cell
def _(ValidationCallback, WandbLogger, dc, metric):
    from deepchem.models import GATModel
    featurizer = dc.feat.MolGraphConvFeaturizer()
    tasks_1, _datasets, transformers_1 = dc.molnet.load_tox21(reload=False, featurizer=featurizer, transformers=[])
    train_dataset_1, valid_dataset_1, test_dataset_1 = _datasets
    wandblogger_2 = WandbLogger(project='deepchem_graphconv', name='GAT')
    model_2 = GATModel(mode='classification', n_tasks=len(tasks_1), batch_size=100, learning_rate=0.001, wandb_logger=wandblogger_2)
    vc_valid_1 = ValidationCallback(valid_dataset_1, interval=100, metrics=[metric], transformers=transformers_1)
    model_2.fit(train_dataset_1, nb_epoch=50, callbacks=[vc_valid_1])
    return (
        model_2,
        test_dataset_1,
        train_dataset_1,
        transformers_1,
        valid_dataset_1,
        wandblogger_2,
    )


@app.cell
def _(
    metric,
    metrics_table,
    model_2,
    test_dataset_1,
    train_dataset_1,
    transformers_1,
    valid_dataset_1,
):
    train_score3 = model_2.evaluate(train_dataset_1, [metric], transformers_1)
    valid_score3 = model_2.evaluate(valid_dataset_1, [metric], transformers_1)
    test_score3 = model_2.evaluate(test_dataset_1, [metric], transformers_1)
    metrics_table.add_data('Graph Attention Network', train_score3, valid_score3, test_score3)
    # Add a row to our table for our GAT
    print('Training set score:', train_score3)
    print('Validation set score:', valid_score3)
    print('Test set score:', test_score3)
    return


@app.cell
def _(metrics_table, wandblogger_2):
    # Log the final table to this run
    wandblogger_2.wandb_run.log({'Scores': metrics_table})
    return


@app.cell
def _(wandblogger_2):
    wandblogger_2.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Congratulations! Time to join the Community!

    Congratulations on completing this tutorial notebook! If you enjoyed working through the tutorial, and want to continue working with DeepChem, we encourage you to finish the rest of the tutorials in this series. You can also help the DeepChem community in the following ways:

    ## Star DeepChem on [GitHub](https://github.com/deepchem/deepchem)
    This helps build awareness of the DeepChem project and the tools for open source drug discovery that we're trying to build.

    ## Join the DeepChem Gitter
    The DeepChem [Gitter](https://gitter.im/deepchem/Lobby) hosts a number of scientists, developers, and enthusiasts interested in deep learning for the life sciences. Join the conversation!
    """)
    return


if __name__ == "__main__":
    app.run()
