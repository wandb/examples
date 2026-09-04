# /// script
# dependencies = ["monai", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/monai/3d_brain_tumor_segmentation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Brain tumor 3D segmentation with MONAI and Weights & Biases

    [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wandb/examples/blob/main/colabs/monai/3d_brain_tumor_segmentation.ipynb)

    This tutorial shows how to construct a training workflow of multi-labels 3D brain tumor segmentation task using [MONAI](https://github.com/Project-MONAI/MONAI) and use experiment tracking and data visualization features of [Weights & Biases](https://wandb.ai/site). The tutorial contains the following features:

    1. Initialize a Weights & Biases run and synchrozize all configs associated with the run for reproducibility.
    2. MONAI transform API:
        1. MONAI Transforms for dictionary format data.
        2. How to define a new transform according to MONAI `transforms` API.
        3. How to randomly adjust intensity for data augmentation.
    3. Data Loading and Visualization:
        1. Load Nifti image with metadata, load a list of images and stack them.
        2. Cache IO and transforms to accelerate training and validation.
        3. Visualize the data using `wandb.Table` and interactive segmentation overlay on Weights & Biases.
    4. Training a 3D `SegResNet` model
        1. Using the `networks`, `losses`, and `metrics` APIs from MONAI.
        2. Training the 3D `SegResNet` model using a PyTorch training loop.
        3. Track the training experiment using Weights & Biases.
        4. Log and version model checkpoints as model artifacts on Weights & Biases.
    5. Visualize and compare the predictions on the validation dataset using `wandb.Table` and interactive segmentation overlay on Weights & Biases.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🌴 Setup and Installation

    First, let us install the latest version of both MONAI and Weights and Biases.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: monai[nibabel, tqdm] !python -c "import monai" || pip install -q -U "monai[nibabel, tqdm]"
    # packages added via marimo's package management: wandb !python -c "import wandb" || pip install -q -U wandb
    return


@app.cell
def _():
    import os

    import numpy as np
    from tqdm.auto import tqdm
    import wandb

    from monai.apps import DecathlonDataset
    from monai.data import DataLoader, decollate_batch
    from monai.losses import DiceLoss
    from monai.config import print_config
    from monai.inferers import sliding_window_inference
    from monai.metrics import DiceMetric
    from monai.networks.nets import SegResNet
    from monai.transforms import (
        Activations,
        AsDiscrete,
        Compose,
        LoadImaged,
        MapTransform,
        NormalizeIntensityd,
        Orientationd,
        RandFlipd,
        RandScaleIntensityd,
        RandShiftIntensityd,
        RandSpatialCropd,
        Spacingd,
        EnsureTyped,
        EnsureChannelFirstd,
    )
    from monai.utils import set_determinism

    import torch

    print_config()
    return (
        Activations,
        AsDiscrete,
        Compose,
        DataLoader,
        DecathlonDataset,
        DiceLoss,
        DiceMetric,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        MapTransform,
        NormalizeIntensityd,
        Orientationd,
        RandFlipd,
        RandScaleIntensityd,
        RandShiftIntensityd,
        RandSpatialCropd,
        SegResNet,
        Spacingd,
        decollate_batch,
        np,
        os,
        set_determinism,
        sliding_window_inference,
        torch,
        tqdm,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will then authenticate this colab instance to use W&B.
    """)
    return


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🌳 Initialize a W&B Run

    We will start a new W&B run to start tracking our experiment.
    """)
    return


@app.cell
def _(wandb):
    wandb.init(project="monai-brain-tumor-segmentation")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use of proper config system is a recommended best practice for reproducible machine learning. We can track the hyperparameters for every experiment using W&B.
    """)
    return


@app.cell
def _(wandb):
    config = wandb.config
    config.seed = 0
    config.roi_size = [224, 224, 144]
    config.batch_size = 1
    config.num_workers = 4
    config.max_train_images_visualized = 20
    config.max_val_images_visualized = 20
    config.dice_loss_smoothen_numerator = 0
    config.dice_loss_smoothen_denominator = 1e-5
    config.dice_loss_squared_prediction = True
    config.dice_loss_target_onehot = False
    config.dice_loss_apply_sigmoid = True
    config.initial_learning_rate = 1e-4
    config.weight_decay = 1e-5
    config.max_train_epochs = 50
    config.validation_intervals = 1
    config.dataset_dir = "./dataset/"
    config.checkpoint_dir = "./checkpoints"
    config.inference_roi_size = (128, 128, 64)
    config.max_prediction_images_visualized = 20
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We would also need to set the random seed for modules to enable or disable deterministic training.
    """)
    return


@app.cell
def _(config, os, set_determinism):
    set_determinism(seed=config.seed)

    # Create directories
    os.makedirs(config.dataset_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 💿 Data Loading and Transformation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we use the `monai.transforms` API to create a custom transform that converts the multi-classes labels into multi-labels segmentation task in one-hot format.
    """)
    return


@app.cell
def _(MapTransform, torch):
    class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
        """
        Convert labels to multi channels based on brats classes:
        label 1 is the peritumoral edema
        label 2 is the GD-enhancing tumor
        label 3 is the necrotic and non-enhancing tumor core
        The possible classes are TC (Tumor core), WT (Whole tumor)
        and ET (Enhancing tumor).

        Reference: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

        """

        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                result = []
                # merge label 2 and label 3 to construct TC
                result.append(torch.logical_or(d[key] == 2, d[key] == 3))
                # merge labels 1, 2 and 3 to construct WT
                result.append(
                    torch.logical_or(
                        torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                    )
                )
                # label 2 is ET
                result.append(d[key] == 2)
                d[key] = torch.stack(result, axis=0).float()
            return d

    return (ConvertToMultiChannelBasedOnBratsClassesd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we set up transforms for training and validation datasets respectively.
    """)
    return


@app.cell
def _(
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    config,
):
    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=config.roi_size, random_size=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    return train_transform, val_transform


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🍁 The Dataset

    The dataset that we will use for this experiment comes from http://medicaldecathlon.com/. We will use Multimodal multisite MRI data (FLAIR, T1w, T1gd, T2w) to segment Gliomas, necrotic/active tumour, and oedema. The dataset consists of 750 4D volumes (484 Training + 266 Testing).

    We will use the `DecathlonDataset` to automatically download and extract the dataset. It inherits MONAI `CacheDataset` which enables us to set `cache_num=N` to cache `N` items for training and use the default args to cache all the items for validation, depending on your memory size.
    """)
    return


@app.cell
def _(DecathlonDataset, config, val_transform):
    train_dataset = DecathlonDataset(
        root_dir=config.dataset_dir,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="training",
        download=True,
        cache_rate=0.0,
        num_workers=4,
    )
    val_dataset = DecathlonDataset(
        root_dir=config.dataset_dir,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=4,
    )
    return train_dataset, val_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note:** Instead of applying the `train_transform` to the `train_dataset`, we have applied `val_transform` to both the training and validation datasets. This is because, before training, we would be visualizing samples from both the splits of the dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 📸 Visualizing the Dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Weights & Biases supports images, video, audio, and more. Log rich media to explore our results and visually compare our runs, models, and datasets. We would be using the [segmentation mask overlay system](https://docs.wandb.ai/guides/track/log/media#image-overlays-in-tables) to visualize our data volumes. To log segmentation masks in [tables](https://docs.wandb.ai/guides/tables), we will need to provide a `wandb.Image`` object for each row in the table.

    An example is provided in the Code snippet below:

    ```python
    table = wandb.Table(columns=["ID", "Image"])

    for id, img, label in zip(ids, images, labels):
        mask_img = wandb.Image(
            img,
            masks={
                "prediction": {"mask_data": label, "class_labels": class_labels}
                # ...
            },
        )

        table.add_data(id, img)

    wandb.log({"Table": table})
    ```

    Let us now write a simple utility function that takes a sample image, label, `wandb.Table` object and some associated metadata and populate the rows of a table that would be logged to our Weights & Biases dashboard.
    """)
    return


@app.cell
def _(np, tqdm, wandb):
    def log_data_samples_into_tables(sample_image: np.array, sample_label: np.array, split: str=None, data_idx: int=None, table: wandb.Table=None):
        num_channels, _, _, num_slices = sample_image.shape
        with tqdm(total=num_slices, leave=False) as _progress_bar:
            for slice_idx in range(num_slices):
                ground_truth_wandb_images = []
                for channel_idx in range(num_channels):
                    ground_truth_wandb_images.append(wandb.Image(sample_image[channel_idx, :, :, slice_idx], masks={'ground-truth/Tumor-Core': {'mask_data': sample_label[0, :, :, slice_idx], 'class_labels': {0: 'background', 1: 'Tumor Core'}}, 'ground-truth/Whole-Tumor': {'mask_data': sample_label[1, :, :, slice_idx] * 2, 'class_labels': {0: 'background', 2: 'Whole Tumor'}}, 'ground-truth/Enhancing-Tumor': {'mask_data': sample_label[2, :, :, slice_idx] * 3, 'class_labels': {0: 'background', 3: 'Enhancing Tumor'}}}))
                table.add_data(split, _data_idx, slice_idx, *ground_truth_wandb_images)
                _progress_bar.update(1)
        return table

    return (log_data_samples_into_tables,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we define the `wandb.Table` object and what columns it consists of so that we can populate with our data visualizations.
    """)
    return


@app.cell
def _(wandb):
    table = wandb.Table(
        columns=[
            "Split",
            "Data Index",
            "Slice Index",
            "Image-Channel-0",
            "Image-Channel-1",
            "Image-Channel-2",
            "Image-Channel-3",
        ]
    )
    return (table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then we loop over the `train_dataset` and `val_dataset` respectively to generate the visualizations for the data samples and populate the rows of the table which we would log to our dashboard.
    """)
    return


@app.cell
def _(
    config,
    log_data_samples_into_tables,
    table,
    tqdm,
    train_dataset,
    val_dataset,
    wandb,
):
    _max_samples = min(config.max_train_images_visualized, len(train_dataset)) if config.max_train_images_visualized > 0 else len(train_dataset)
    _progress_bar = tqdm(enumerate(train_dataset[:_max_samples]), total=_max_samples, desc='Generating Train Dataset Visualizations:')
    for _data_idx, _sample in _progress_bar:
        sample_image = _sample['image'].detach().cpu().numpy()
        sample_label = _sample['label'].detach().cpu().numpy()
        table_1 = log_data_samples_into_tables(sample_image, sample_label, split='train', data_idx=_data_idx, table=table)
    _max_samples = min(config.max_val_images_visualized, len(val_dataset)) if config.max_val_images_visualized > 0 else len(val_dataset)
    _progress_bar = tqdm(enumerate(val_dataset[:_max_samples]), total=_max_samples, desc='Generating Validation Dataset Visualizations:')
    for _data_idx, _sample in _progress_bar:
        sample_image = _sample['image'].detach().cpu().numpy()
        sample_label = _sample['label'].detach().cpu().numpy()
        table_1 = log_data_samples_into_tables(sample_image, sample_label, split='val', data_idx=_data_idx, table=table_1)
    wandb.log({'Tumor-Segmentation-Data': table_1})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data appears to us on our W&B dashboard in an interactive tabular format. We can see each channel of a particular slice from a data volume overlayed with the respective segmentation mask in each row. Let us write [Weave queries](https://docs.wandb.ai/guides/weave) to filter the data on our table and focus on one particular row.

    ![](./assets/viz-1.gif)

    Let us now open an image and check how we can interact with each of the segmentation masks using the interactive overlay.

    ![](./assets/viz-2.gif)

    **Note:** The labels in the dataset consist of non-overlapping masks across classes, hence, they were logged as separate masks in the overlay.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🛫 Loading the Data

    We create the PyTorch dataloaders for loading the data from the datasets. Note that before creating the dataloaders, we set the `transform` for `train_dataset` to `train_transform` to preprocess and transform the data for training.
    """)
    return


@app.cell
def _(DataLoader, config, train_dataset, train_transform, val_dataset):
    # apply train_transforms to the training dataset
    train_dataset.transform = train_transform

    # create the train_loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    # create the val_loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🤖 Creating the Model, Loss, and Optimizer
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this tutorial we will be training a `SegResNet` model based on the paper [3D MRI brain tumor segmentation using autoencoder regularization](https://arxiv.org/pdf/1810.11654.pdf). We create the `SegResNet` model that comes implemented as a PyTorch Module as part of the `monai.networks` API. We also create our optimizer and learning rate scheduler.
    """)
    return


@app.cell
def _(SegResNet, config, torch):
    device = torch.device("cuda:0")

    # create model
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)

    # create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        config.initial_learning_rate,
        weight_decay=config.weight_decay,
    )

    # create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_train_epochs
    )
    return device, lr_scheduler, model, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We define our loss as multi-label `DiceLoss` using the `monai.losses` API and the corresponding dice metrics using the `monai.metrics` API.
    """)
    return


@app.cell
def _(Activations, AsDiscrete, Compose, DiceLoss, DiceMetric, config, torch):
    loss_function = DiceLoss(
        smooth_nr=config.dice_loss_smoothen_numerator,
        smooth_dr=config.dice_loss_smoothen_denominator,
        squared_pred=config.dice_loss_squared_prediction,
        to_onehot_y=config.dice_loss_target_onehot,
        sigmoid=config.dice_loss_apply_sigmoid,
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # use automatic mixed-precision to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    return dice_metric, dice_metric_batch, loss_function, post_trans, scaler


@app.cell
def _(sliding_window_inference, torch):
    def inference(model, input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        with torch.cuda.amp.autocast():
            return _compute(input)

    return (inference,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🚝 Training and Validation

    Before we start training, let us define some metric properties which will later be logged with `wandb.log()` for tracking our training and validation experiments.
    """)
    return


@app.cell
def _(wandb):
    wandb.define_metric("epoch/epoch_step")
    wandb.define_metric("epoch/*", step_metric="epoch/epoch_step")
    wandb.define_metric("batch/batch_step")
    wandb.define_metric("batch/*", step_metric="batch/batch_step")
    wandb.define_metric("validation/validation_step")
    wandb.define_metric("validation/*", step_metric="validation/validation_step")

    batch_step = 0
    validation_step = 0
    metric_values = []
    metric_values_tumor_core = []
    metric_values_whole_tumor = []
    metric_values_enhanced_tumor = []
    return (
        batch_step,
        metric_values,
        metric_values_enhanced_tumor,
        metric_values_tumor_core,
        metric_values_whole_tumor,
        validation_step,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🍭 Execute Standard PyTorch Training Loop
    """)
    return


@app.cell
def _(
    batch_step,
    config,
    decollate_batch,
    device,
    dice_metric,
    dice_metric_batch,
    inference,
    loss_function,
    lr_scheduler,
    metric_values,
    metric_values_enhanced_tumor,
    metric_values_tumor_core,
    metric_values_whole_tumor,
    model,
    optimizer,
    os,
    post_trans,
    scaler,
    torch,
    tqdm,
    train_dataset,
    train_loader,
    val_loader,
    validation_step,
    wandb,
):
    # Define a W&B Artifact object
    artifact = wandb.Artifact(name=f'{wandb.run.id}-checkpoint', type='model')
    epoch_progress_bar = tqdm(range(config.max_train_epochs), desc='Training:')
    for epoch in epoch_progress_bar:
        model.train()
        epoch_loss = 0
        total_batch_steps = len(train_dataset) // train_loader.batch_size
        batch_progress_bar = tqdm(train_loader, total=total_batch_steps, leave=False)
        for batch_data in batch_progress_bar:
            inputs, labels = (batch_data['image'].to(device), batch_data['label'].to(device))
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()  # Training Step
            scaler.step(optimizer)
            scaler.update()
            epoch_loss = epoch_loss + loss.item()
            batch_progress_bar.set_description(f'train_loss: {loss.item():.4f}:')
            wandb.log({'batch/batch_step': batch_step, 'batch/train_loss': loss.item()})
            batch_step_1 = batch_step + 1
        lr_scheduler.step()
        epoch_loss = epoch_loss / total_batch_steps
        wandb.log({'epoch/epoch_step': epoch, 'epoch/mean_train_loss': epoch_loss, 'epoch/learning_rate': lr_scheduler.get_last_lr()[0]})
        epoch_progress_bar.set_description(f'Training: train_loss: {epoch_loss:.4f}:')
        if (epoch + 1) % config.validation_intervals == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (val_data['image'].to(device), val_data['label'].to(device))  ## Log batch-wise training loss to W&B
                    val_outputs = inference(model, val_inputs)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)
                metric_values.append(dice_metric.aggregate().item())
                metric_batch = dice_metric_batch.aggregate()  ## Log batch-wise training loss and learning rate to W&B
                metric_values_tumor_core.append(metric_batch[0].item())
                metric_values_whole_tumor.append(metric_batch[1].item())
                metric_values_enhanced_tumor.append(metric_batch[2].item())
                dice_metric.reset()
                dice_metric_batch.reset()
                checkpoint_path = os.path.join(config.checkpoint_dir, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                artifact.add_file(local_path=checkpoint_path)
                wandb.log_artifact(artifact, aliases=[f'epoch_{epoch}'])
                wandb.log({'validation/validation_step': validation_step, 'validation/mean_dice': metric_values[-1], 'validation/mean_dice_tumor_core': metric_values_tumor_core[-1], 'validation/mean_dice_whole_tumor': metric_values_whole_tumor[-1], 'validation/mean_dice_enhanced_tumor': metric_values_enhanced_tumor[-1]})  # Validation and model checkpointing
                validation_step_1 = validation_step + 1
    # Wait for this artifact to finish logging
    artifact.wait()  # Log and versison model checkpoints using W&B artifacts.  # Log validation metrics to W&B dashboard.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Instrumenting our code with `wandb.log` not only enables us to track all the metrics associated with our training and validation process, but also the all system metrics (our CPU and GPU in this case) on our W&B dashboard.

    ![](./assets/viz-3.gif)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we navigate to the artifacts tab in the W&B run dashboard, we will be able to access the different versions of model checkpoint artifacts that we logged during training.

    ![](./assets/viz-4.gif)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🔱 Inference
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using the artifacts interface, we can select which version of the artifact is the best model checkpoint, in this case, the mean epoch-wise training loss. We can also explore the entire lineage of the artifact and also use the version that we need.

    ![](./assets/viz-5.gif)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us fetch the version of the model artifact with the best epoch-wise mean training loss and load the checkpoint state dictionary to the model.
    """)
    return


@app.cell
def _(model, os, torch, wandb):
    model_artifact = wandb.use_artifact(
        "geekyrakshit/monai-brain-tumor-segmentation/d5ex6n4a-checkpoint:v49",
        type="model",
    )
    model_artifact_dir = model_artifact.download()
    model.load_state_dict(torch.load(os.path.join(model_artifact_dir, "model.pth")))
    model.eval()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 📸 Visualizing Predictions and Comparing with the Ground Truth Labels

    In order to visualize the predictions of the pre-trained model and compare them with the corresponding ground-truth segmentation mask using the interactive segmentation mask overlay, let us create another ultility function.
    """)
    return


@app.cell
def _(np, tqdm, wandb):
    def log_predictions_into_tables(sample_image: np.array, sample_label: np.array, predicted_label: np.array, split: str=None, data_idx: int=None, table: wandb.Table=None):
        num_channels, _, _, num_slices = sample_image.shape
        with tqdm(total=num_slices, leave=False) as _progress_bar:
            for slice_idx in range(num_slices):
                wandb_images = []
                for channel_idx in range(num_channels):
                    wandb_images = wandb_images + [wandb.Image(sample_image[channel_idx, :, :, slice_idx], masks={'ground-truth/Tumor-Core': {'mask_data': sample_label[0, :, :, slice_idx], 'class_labels': {0: 'background', 1: 'Tumor Core'}}, 'prediction/Tumor-Core': {'mask_data': predicted_label[0, :, :, slice_idx] * 2, 'class_labels': {0: 'background', 2: 'Tumor Core'}}}), wandb.Image(sample_image[channel_idx, :, :, slice_idx], masks={'ground-truth/Whole-Tumor': {'mask_data': sample_label[1, :, :, slice_idx], 'class_labels': {0: 'background', 1: 'Whole Tumor'}}, 'prediction/Whole-Tumor': {'mask_data': predicted_label[1, :, :, slice_idx] * 2, 'class_labels': {0: 'background', 2: 'Whole Tumor'}}}), wandb.Image(sample_image[channel_idx, :, :, slice_idx], masks={'ground-truth/Enhancing-Tumor': {'mask_data': sample_label[2, :, :, slice_idx], 'class_labels': {0: 'background', 1: 'Enhancing Tumor'}}, 'prediction/Enhancing-Tumor': {'mask_data': predicted_label[2, :, :, slice_idx] * 2, 'class_labels': {0: 'background', 2: 'Enhancing Tumor'}}})]
                table.add_data(split, _data_idx, slice_idx, *wandb_images)
                _progress_bar.update(1)
        return table

    return (log_predictions_into_tables,)


@app.cell
def _(
    config,
    device,
    inference,
    log_predictions_into_tables,
    model,
    post_trans,
    torch,
    tqdm,
    val_dataset,
    wandb,
):
    # create the prediction table
    prediction_table = wandb.Table(columns=['Split', 'Data Index', 'Slice Index', 'Image-Channel-0/Tumor-Core', 'Image-Channel-1/Tumor-Core', 'Image-Channel-2/Tumor-Core', 'Image-Channel-3/Tumor-Core', 'Image-Channel-0/Whole-Tumor', 'Image-Channel-1/Whole-Tumor', 'Image-Channel-2/Whole-Tumor', 'Image-Channel-3/Whole-Tumor', 'Image-Channel-0/Enhancing-Tumor', 'Image-Channel-1/Enhancing-Tumor', 'Image-Channel-2/Enhancing-Tumor', 'Image-Channel-3/Enhancing-Tumor'])
    with torch.no_grad():
        config.max_prediction_images_visualized
        _max_samples = min(config.max_prediction_images_visualized, len(val_dataset)) if config.max_prediction_images_visualized > 0 else len(val_dataset)
        _progress_bar = tqdm(enumerate(val_dataset[:_max_samples]), total=_max_samples, desc='Generating Predictions:')
        for _data_idx, _sample in _progress_bar:
            val_input = _sample['image'].unsqueeze(0).to(device)
            val_output = inference(model, val_input)
            val_output = post_trans(val_output[0])
            prediction_table = log_predictions_into_tables(sample_image=_sample['image'].cpu().numpy(), sample_label=_sample['label'].cpu().numpy(), predicted_label=val_output.cpu().numpy(), data_idx=_data_idx, split='validation', table=prediction_table)
        wandb.log({'Predictions/Tumor-Segmentation-Data': prediction_table})
    # Perform inference and visualization
    # End the experiment
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us see how we can analyze and compare the predicted segmentation masks and the ground-truth labels for each class using the interactive segmentation mask overlay.

    ![](./assets/viz-6.gif)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also check out the report [Brain Tumor Segmentation using MONAI and WandB](https://wandb.ai/geekyrakshit/brain-tumor-segmentation/reports/Brain-Tumor-Segmentation-using-MONAI-and-WandB---Vmlldzo0MjUzODIw) for more details regarding training a brain-tumor segmentation model using MONAI and W&B.
    """)
    return


if __name__ == "__main__":
    app.run()
