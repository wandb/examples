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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Dataset_Visualization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    _W&B Datasets & Predictions is currently in the early-access phase. You can use it in our production service at [wandb.ai](https://wandb.ai), with [some limitations](https://docs.wandb.com/datasets-and-predictions#current-limitations). APIs are subject to change. We'd love to hear questions, comments, and ideas! Drop us a line at feedback@wandb.com._

    # WandB Dataset Visualization Demo

    This notebook demonstrates WandB's dataset visualization features. In particular we will show how WandB [Artifacts](https://docs.wandb.com/artifacts) can be used to visualize datasets and predictions, with a focus on image data. We will track model and data lineage as well as perform interactive model analysis on the resulting datasets. The overall flow will be:

    1. Create a dataset
    2. Split the dataset into train and test
    3. Train a model to make predictions on the transformed dataet
    4. Log predications from the model against training and evaluation sets
    5. Analyze the model in WandB's UI
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 0: Setup

    ## Install requirements & utils

    For brevity, we put utility functions for working with the dataset in `util.py`.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # # Install the python dependencies
    # !pip install matplotlib numpy Pillow wandb
    # 
    # # Download a util file of helper methods for this notebook
    # !curl https://raw.githubusercontent.com/wandb/dsviz-demo/master/util.py --output util.py
    return


@app.cell
def _():
    # Colab sometimes has problems with Pillow. If you are facing this issue, 
    # uncomment the `exit()` line and run this cell. Then rerun the `!pip install`
    # cell above.

    # exit()
    return


@app.cell
def _():
    import util
    import matplotlib.pyplot as plt
    from PIL import Image
    import os
    import wandb
    print(wandb.__version__)
    return os, plt, util, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Login to wandb
    """)
    return


@app.cell
def _():
    # default project name where results will be logged
    WANDB_PROJECT = "dsviz-demo-colab"
    NUM_EXAMPLES = 50
    return NUM_EXAMPLES, WANDB_PROJECT


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download the data

    Before we get started, we will download an example dataset to our local machine. This is a big dataset, so please be patient if you are on a slow connection. For brevity, we put utility functions for working with the dataset in `util.py`. After the download is complete, we will show an example of the data.

    **Note:** if you see the error "``AttributeError: module 'PIL.TiffTags' has no attribute 'IFD'``", this is likely a [Colab issue](https://github.com/facebookresearch/detectron2/issues/2231) which can be solved by restarting your runtime (header menu > Runtime > Restart runtime).
    """)
    return


@app.cell
def _(util):
    # Download the data if not already present
    util.download_data()
    # Show an example training image
    util.show_image(util.get_train_image_path(0))
    # Show an example of color mask
    util.show_image(util.get_color_label_image_path(0))

    # Print the label types:
    print("Class Mapping:")
    print(list(zip(util.BDD_IDS, util.BDD_CLASSES)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 1: Build the dataset

    First, let's build a dataset for use in the rest of this project. We will do this in the context of a `wandb.Run`. A `Run` is an isolated process which can optionally depend on upstream artifacts as well as optionally produce artifacts for later consumption. In this step, we will create a `wandb.Table` during our run and output it in an artifact. This table will contain all of our raw data for later use. Moreover, W&B offers rich tools to analyze and visualize such Tables in the interactive UI.
    """)
    return


@app.cell
def _(NUM_EXAMPLES, WANDB_PROJECT, util, wandb):
    # Initialize the run
    with wandb.init(project=WANDB_PROJECT, job_type='create_dataset', config={'num_examples': NUM_EXAMPLES, 'scale_factor': 2}) as _run:
        class_set = wandb.Classes([{'name': name, 'id': id} for name, id in zip(util.BDD_CLASSES, util.BDD_IDS)])  # The project to register this Run to
        table = wandb.Table(columns=['id', 'train_image', 'colored_image', 'label_mask', 'dominant_class'])  # The type of this Run. Runs of the same type can be grouped together in the UI
        for ndx in range(_run.config['num_examples']):  # Custom configuration parameters which you might want to tune or adjust for the Run
            example = wandb.Image(util.get_scaled_train_image(ndx, _run.config.scale_factor), classes=class_set, masks={'ground_truth': {'mask_data': util.get_scaled_mask_label(ndx, _run.config.scale_factor)}}, boxes={'ground_truth': {'box_data': util.get_scaled_bounding_boxes(ndx, _run.config.scale_factor)}})  # The number of raw samples to include.
            color_label = wandb.Image(util.get_scaled_color_mask(ndx, _run.config.scale_factor))  # The scaling factor for the images
            label_mask = wandb.Image(util.get_scaled_mask_label(ndx, _run.config.scale_factor))
            table.add_data(util.train_ids[ndx], example, color_label, label_mask, util.get_dominant_class(label_mask))
        _artifact = wandb.Artifact(name='raw_data', type='dataset')  # Setup a WandB Classes object. This will give additional metadata for visuals
        _artifact.add(table, 'raw_examples')
        _run.log_artifact(_artifact)
        print('Saving data to WandB...')
    print('... Run Complete')  # Setup a WandB Table object to hold our dataset  # Fill up the table  # First, we will build a wandb.Image to act as our raw example object  #    classes: the classes which map to masks and/or box metadata  #    masks: the mask metadata. In this case, we use a 2d array where each cell corresponds to the label (this comes directly from the dataset)  #    boxes: the bounding box metadata. For example sake, we create bounding boxes by looking at the mask data and creating boxes which fully enclose each class.  #           The data is an array of objects like:  #                 "position": {  #                             "minX": minX,  #                             "maxX": maxX,  #                             "minY": minY,  #                             "maxY": maxY,  #                         },  #                         "class_id" : id_num,  #                     }  # Next, we create two additional images which may be helpful during analysis. Notice that the additional metadata is optional.  # Finally, we add a row of our newly constructed data.  # Create an Artifact (versioned folder)  # .add the table to the artifact  # Finally, log the artifact
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Review the dataset in the Dashboard

    Great, now if you click on the URL above, you should land on a run page. Since we did not log any metrics, there are no charts. Click the database icon (it looks like a stack of hockey pucks) on the left panel to see this run's artifacts. You should see something similar to the following:

    ![Raw Data](https://raw.githubusercontent.com/wandb/dsviz-demo/master/notebook_images/raw_data.png)

    Click on the "`raw_data`" row and navigate to the "Files" table. It should look like this:

    ![Raw Data](https://raw.githubusercontent.com/wandb/dsviz-demo/master/notebook_images/raw_data_files.png)

    Clicking on the "`raw_examples.table.json`" entry will launch an interactive data explorer to review the table we just built:

    ![Raw Data](https://raw.githubusercontent.com/wandb/dsviz-demo/master/notebook_images/raw_data_explore.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 2: Splitting the data into train and test

    Next, we will split the data into a train and a test dataset. Similar to before, we will launch a `Run` to perform this operation. Remember, this new execution could happen on a different machine as we will dynamically load the needed resources. In particular, we will lood in the raw dataset from the last run, and output 2 new datasets.
    """)
    return


@app.cell
def _(WANDB_PROJECT, os, plt, wandb):
    # This step should look familiar by now:
    with wandb.init(project=WANDB_PROJECT, job_type='split_dataset', config={'train_pct': 0.7}) as _run:
        _dataset_artifact = _run.use_artifact('raw_data:latest')
        _data_table = _dataset_artifact.get('raw_examples')
        print('\nExample Data row\n', _data_table.data[0])
        print('\nExample Image\n')
        plt.imshow(_data_table.data[0][1]._image)
        plt.show()
        print('\nArtifact Directory Contents: \n', os.listdir('artifacts'))
        train_count = int(len(_data_table.data) * _run.config.train_pct)  # Get the latest version of the artifact. Notice the name alias follows this convention: "<ARTIFACT_NAME>:<VERSION>"
        _train_table = wandb.Table(columns=_data_table.columns, data=_data_table.data[:train_count])  # When version is set to "latest", then the latest version will always be used.
        _test_table = wandb.Table(columns=_data_table.columns, data=_data_table.data[train_count:])  # However, you can pin to a version by using an alias such as "raw_data:v0"
        _train_artifact = wandb.Artifact('train_data', 'dataset')
        _test_artifact = wandb.Artifact('test_data', 'dataset')
        _train_artifact.add(_train_table, 'train_table')  # Next, we .get the table by the same name that we saved it in the last run.
        _test_artifact.add(_test_table, 'test_table')
        _run.log_artifact(_train_artifact)
        _run.log_artifact(_test_artifact)  # Print a row
        print('Saving data to WandB...')
    print('... Run Complete')  # Show an example image  # Notice that a new directory was made: artifacts which is managed by wandb  # Now we can build two separate artifacts for later use. We will first split the raw table into two parts,  # then create two different artifacts, each of which will hold our new tables. We create two artifacts so that  # in future runs, we can selectively decide which subsets of data to download.  # Create the tables  # Create the artifacts  # Save the tables to the artifacts with .add  # Log the artifacts out as outputs of the run
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Review the splits in the Dashboard
    Notice, in this step, the raw_data `wandb.Table` was reinstatiated and the data, images, etc... came along for the ride. This makes it easy for ML practitioners on a team to share data and assets easily. To manage this, you can see that we created an artifacts directory to save local data.

    Now we have two new datasets. Feel free to browse them similar to our last step. However, this time, click "Graph View" rather than "Files" to see the lineage of the artifact:

    ![](https://raw.githubusercontent.com/wandb/dsviz-demo/master/notebook_images/split_data.png)

    ![](https://raw.githubusercontent.com/wandb/dsviz-demo/master/notebook_images/split_graph.png)

    We will come back to this graph view later on!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 3: Model Training

    Now we will train a model to predict bounding boxes. For the sake of simplicity, we will "train" a model which splits the image into it's grayscale quantiles and assigns labels to each patch. As you can imagine, the model performance can be improved dramatically.
    """)
    return


@app.cell
def _(WANDB_PROJECT, util, wandb):
    # Again, create a run.
    with wandb.init(project=WANDB_PROJECT, job_type='model_train') as _run:
        _train_artifact = _run.use_artifact('train_data:latest')
        _train_table = _train_artifact.get('train_table')  # Similar to before, we will load in the artifact and asset we need. In this case, the training data
        train_data, _mask_data = util.make_datasets(_train_table, util.n_classes)
        _model = util.ExampleSegmentationModel(util.n_classes)
        _model.train(train_data, _mask_data)
        _scores, _results = util.score_model(_model, train_data, _mask_data, util.n_classes)  # Next, we split out the labels and train the model
        results_table = wandb.Table(columns=['id', 'pred_mask', 'dominant_pred'] + util.BDD_CLASSES, data=[[_train_table.data[ndx][0], wandb.Image(_train_table.data[ndx][1], masks={'train_predicted_truth': {'mask_data': _results[ndx]}}, boxes={'ground_truth': {'box_data': util.mask_to_bounding(_results[ndx])}}), util.BDD_CLASSES[util.get_dominant_id_ndx(_results[ndx])]] + list(row) for ndx, row in enumerate(_scores)])
        _results_artifact = wandb.Artifact('train_results', 'dataset')
        _results_artifact.add(results_table, 'train_iou_score_table')
        _run.log_artifact(_results_artifact)
        _model.save('model.pkl')  # Finally we score the model. Behind the scenes, we score each mask on its IOU score.
        _model_artifact = wandb.Artifact('trained_model', 'model')
        _model_artifact.add_file('model.pkl')
        _run.log_artifact(_model_artifact)  # Let's create a new table. Notice that we create many columns - an evaluation score for each class type.
        print('Saving data to WandB...')
    print('... Run Complete')  # Data construction is similar to before, but we now use the predicted masks and bound boxes.  # We create an artifact, add the table, and log it as part of the run.  # Finally, let's save the model as a flat file and add that to its own artifact.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 4: Model Evaluation

    Now that we have a trained model, we want to score it on the test data which was held out in step 2. This code is very similar to the training step, with the execption of slightly different naming. The important difference is that we load the saved model from the artifact.
    """)
    return


@app.cell
def _(WANDB_PROJECT, util, wandb):
    with wandb.init(project=WANDB_PROJECT, job_type='model_eval') as _run:
        _test_artifact = _run.use_artifact('test_data:latest')
        _test_table = _test_artifact.get('test_table')  # Retrieve the test data
        test_data, _mask_data = util.make_datasets(_test_table, util.n_classes)
        _model_artifact = _run.use_artifact('trained_model:latest')
        path = _model_artifact.get_path('model.pkl').download()
        _model = util.ExampleSegmentationModel.load(path)
        _scores, _results = util.score_model(_model, test_data, _mask_data, util.n_classes)  # Download the saved model file.
        _results_artifact = wandb.Artifact('test_results', 'dataset')
        data = [[_test_table.data[ndx][0], wandb.Image(_test_table.data[ndx][1], masks={'test_predicted_truth': {'mask_data': _results[ndx]}}, boxes={'ground_truth': {'box_data': util.mask_to_bounding(_results[ndx])}}), util.BDD_CLASSES[util.get_dominant_id_ndx(_results[ndx])]] + list(row) for ndx, row in enumerate(_scores)]
        _results_artifact.add(wandb.Table(['id', 'pred_mask_test', 'dominant_pred_test'] + util.BDD_CLASSES, data=data), 'test_iou_score_table')
        _run.log_artifact(_results_artifact)  # Load the model from the file and score it
        print('Saving data to WandB...')
    print('... Run Complete')  # Create a predicted score table similar to step 3.  # And log out the results.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 5: Model Analysis

    This is where it all comes together. In this step, we join the train and test scoring results with the original dataset and output corresponding artifacts. The new idea introduced here is a `wandb.JoinedTable` which allows you to join two `Table`s for further analysis in the UI.
    """)
    return


@app.cell
def _(WANDB_PROJECT, wandb):
    with wandb.init(project=WANDB_PROJECT, job_type='model_result_analysis') as _run:
        _dataset_artifact = _run.use_artifact('raw_data:latest')
        _data_table = _dataset_artifact.get('raw_examples')  # Retrieve the original raw dataset
        _train_artifact = _run.use_artifact('train_results:latest')
        _train_table = _train_artifact.get('train_iou_score_table')
        _test_artifact = _run.use_artifact('test_results:latest')
        _test_table = _test_artifact.get('test_iou_score_table')  # Retrieve the train and test score tables
        train_results = wandb.JoinedTable(_train_table, _data_table, 'id')
        test_results = wandb.JoinedTable(_test_table, _data_table, 'id')
        _artifact = wandb.Artifact('summary_results', 'dataset')
        _artifact.add(train_results, 'train_results')
        _artifact.add(test_results, 'test_results')
        _run.log_artifact(_artifact)
        print('Saving data to WandB...')  # Join the tables on ID column and log them as outputs.
    print('... Run Complete')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Review the model analysis in the Dashboard
    Now, click on the above **Project** page (second link). This will look like the following:

    ![](https://raw.githubusercontent.com/wandb/dsviz-demo/master/notebook_images/project_page.png)

    Click on the database icon, as previously, to see the artifacts. This time, you are seeing the artifacts for the entire project, with counts of their versions:

    ![](https://raw.githubusercontent.com/wandb/dsviz-demo/master/notebook_images/project_artifacts.png)

    Go ahead and click the "`model`" artifact type, "Files", and "`model.pkl`". The viewer will provide different renderings based on the file type. For a pickled class, you get the following image. For deep networks saved as `.h5` files, you can see all the layers and their attributes.

    ![](https://raw.githubusercontent.com/wandb/dsviz-demo/master/notebook_images/model_view.png)

    Next, head back to the artifact page, click Database type, expand `summary_results`, and select your most recent version. Click "Files" and select one of the join tables:

    ![](https://raw.githubusercontent.com/wandb/dsviz-demo/master/notebook_images/join_view.png)

    Exploring a bit, you can toggle the bounding boxes, masks, group, filter, and sort the data:

    ![](https://raw.githubusercontent.com/wandb/dsviz-demo/master/notebook_images/summary_join.png)

    ![](https://raw.githubusercontent.com/wandb/dsviz-demo/master/notebook_images/grouped.png)

    Finally, click graph view, and "explode". Now, you can visualize the entire process end-to-end:

    ![](https://raw.githubusercontent.com/wandb/dsviz-demo/master/notebook_images/summary_graph.png)
    """)
    return


if __name__ == "__main__":
    app.run()
