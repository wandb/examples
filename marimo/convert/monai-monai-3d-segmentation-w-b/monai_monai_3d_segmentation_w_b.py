# /// script
# dependencies = ["matplotlib", "monai-weekly", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/monai/MONAI_3D_Segmentation_W&B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{monai-example} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{monai-example} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using MONAI and wandb

    ## Introduction
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This tutorial shows how to integrate MONAI into an existing PyTorch medical DL program and use Weights & Biases for experiment tracking.

    This tutorial is modified from a tutorial from MONAI's official GitHub Repository: [Link](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d_visualization_basic.ipynb)

    And easily use below features from MONAI:

    - Transforms for dictionary format data.
    - Load Nifti image with metadata.
    - Add channel dim to the data if no channel dimension.
    - Scale medical image intensity with expected range.
    - Crop out a batch of balanced images based on positive / negative label ratio.
    - Cache IO and transforms to accelerate training and validation.
    - 3D UNet model, Dice loss function, Mean Dice metric for 3D segmentation task.
    - Sliding window inference method.
    - Deterministic training for reproducibility.
    - The Spleen dataset can be downloaded from http://medicaldecathlon.com/.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup Environment
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: monai-weekly[gdown, nibabel, tqdm, ignite] !pip install -q "monai-weekly[gdown, nibabel, tqdm, ignite]"
    # packages added via marimo's package management: wandb !pip install -q wandb
    # packages added via marimo's package management: matplotlib !pip install -q matplotlib
    return


@app.cell
def _():
    import os
    import glob
    import shutil
    import tempfile

    import wandb
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import matplotlib.pyplot as plt

    from monai.utils import first, set_determinism
    from monai.transforms import (
        AsDiscrete,
        AsDiscreted,
        EnsureChannelFirstd,
        Compose,
        CropForegroundd,
        LoadImaged,
        Orientationd,
        RandCropByPosNegLabeld,
        SaveImaged,
        ScaleIntensityRanged,
        Spacingd,
        Invertd,
    )
    from monai.handlers.utils import from_engine
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    from monai.metrics import DiceMetric
    from monai.losses import DiceLoss
    from monai.inferers import sliding_window_inference
    from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
    from monai.config import print_config
    from monai.apps import download_and_extract

    return (
        AsDiscrete,
        CacheDataset,
        Compose,
        CosineAnnealingLR,
        CropForegroundd,
        DataLoader,
        Dataset,
        DiceLoss,
        DiceMetric,
        EnsureChannelFirstd,
        LoadImaged,
        Norm,
        Orientationd,
        RandCropByPosNegLabeld,
        ScaleIntensityRanged,
        Spacingd,
        UNet,
        decollate_batch,
        download_and_extract,
        first,
        glob,
        os,
        plt,
        print_config,
        set_determinism,
        sliding_window_inference,
        tempfile,
        torch,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's print configuration of some packages by using a utility function provided by MONAI as `print_config()` which basically lists down all the versions of the useful libraries.
    """)
    return


@app.cell
def _(print_config):
    print_config()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup data directory
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can specify a directory with the `MONAI_DATA_DIRECTORY` environment variable.
    This allows you to save results and reuse downloads.
    If not specified a temporary directory will be used.
    """)
    return


@app.cell
def _(os, tempfile):
    # set the environment variable
    os.environ["MONAI_DATA_DIRECTORY"] = "./output"
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)
    return (root_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download the dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Downloads and extracts the dataset.
    The dataset comes from http://medicaldecathlon.com/.

    The website has many types of medical datasets like brain tumor, pancreas, heart, prostrate, etc. Here, we are going to use the spleen dataset.

    The spleen is a fist-sized organ in the upper left side of your abdomen, next to your stomach and behind your left ribs.

    It's an important part of your immune system, but you can survive without it. This is because the liver can take over many of the spleen's functions.

    To read more about spleen you can visit [this website](https://www.nhs.uk/conditions/spleen-problems-and-spleen-removal)

    First, we will download the data by specifying the link of the data from the website. Furthermore, we will use a hash value to validate the downloaded file. Finally, we will extract the .tar file. Note, how easy it is to do all of the above steps using the function `download_and_extract`
    """)
    return


@app.cell
def _(download_and_extract, os, root_dir):
    # define the link of the dataset
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    # define the hash value to validate the downloaded file
    md5 = "410d4a301da4e5b2f6f86ec3ddba524e"
    # define the path for downloading the .tar file
    compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
    # define the directory for extracting the contents of the .tar file
    data_dir = os.path.join(root_dir, "Task09_Spleen")
    if not os.path.exists(data_dir):
        # download, extract and validate the file
        download_and_extract(resource, compressed_file, root_dir, md5)
    return (data_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Set MSD Spleen dataset path
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will store the image path and label path as a key value pair in a dictionary and split a subset of data for validation.
    """)
    return


@app.cell
def _(data_dir, glob, os):
    train_images = sorted(
        glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]
    return train_files, val_files


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Set deterministic training for reproducibility
    """)
    return


@app.cell
def _(set_determinism):
    set_determinism(seed=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup transforms for training and validation

    Here we use several transforms to augment the dataset:
    1. `LoadImaged` loads the spleen CT images and labels from NIfTI format files.
    1. `EnsureChannelFirstd` ensures the original data to construct "channel first" shape.
    1. `Orientationd` unifies the data orientation based on the affine matrix.
    1. `Spacingd` adjusts the spacing by `pixdim=(1.5, 1.5, 2.)` based on the affine matrix.
    1. `ScaleIntensityRanged` extracts intensity range [-57, 164] and scales to [0, 1].
    1. `CropForegroundd` removes all zero borders to focus on the valid body area of the images and labels.
    1. `RandCropByPosNegLabeld` randomly crop patch samples from big image based on pos / neg ratio.
    The image centers of negative samples must be in valid body area.
    1. `RandAffined` efficiently performs `rotate`, `scale`, `shear`, `translate`, etc. together based on PyTorch affine transform.
    """)
    return


@app.cell
def _(
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(96, 96, 96),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ]
    )
    return train_transforms, val_transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Check DataLoader
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we will plot a single slice from the first 3D image from the dataloader along with it's label to see if it is loaded and transformed correctly.
    """)
    return


@app.cell
def _(DataLoader, Dataset, first, plt, val_files, val_transforms):
    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image, _label = (check_data['image'][0][0], check_data['label'][0][0])
    print(f'image shape: {image.shape}, label shape: {_label.shape}')
    # plot the slice [:, :, 80]
    plt.figure('check', (12, 6))
    plt.subplot(1, 2, 1)
    plt.title('image')
    plt.imshow(image[:, :, 80], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('label')
    plt.imshow(_label[:, :, 80])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Great, now we will create a function which will log all the slices of the 3D image to W&B to visualize them interactively. Furthermore, we will also log the slices with segmentation masks to see the overlayed view of segmentations masks on the slices interactively in the W&B dashboard.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Logging spleen slices to W&B
    """)
    return


@app.cell
def _(DataLoader, Dataset, first, train_files, val_transforms, wandb):
    # utility function for generating interactive image mask from components
    def wb_mask(bg_img, mask):
        return wandb.Image(bg_img, masks={'ground truth': {'mask_data': mask, 'class_labels': {0: 'background', 1: 'mask'}}})

    def log_spleen_slices(total_slices=100):
        wandb_mask_logs = []
        wandb_img_logs = []
        check_ds = Dataset(data=train_files, transform=val_transforms)
        check_loader = DataLoader(check_ds, batch_size=1)
        check_data = first(check_loader)
        image, _label = (check_data['image'][0][0], check_data['label'][0][0])
        for img_slice_no in range(total_slices):
            img = image[:, :, img_slice_no]  # get the first item of the dataloader
            lbl = _label[:, :, img_slice_no]
            wandb_img_logs.append(wandb.Image(img, caption=f'Slice: {img_slice_no}'))
            wandb_mask_logs.append(wb_mask(img, lbl))
        wandb.log({'Image': wandb_img_logs})
        wandb.log({'Segmentation mask': wandb_mask_logs})  # append the image to wandb_img_list to visualize  # the slices interactively in W&B dashboard  # append the image and masks to wandb_mask_logs  # to see the masks overlayed on the original image

    return (log_spleen_slices,)


@app.cell
def _(log_spleen_slices, wandb):
    # 🐝 init wandb with appropiate project and run name
    wandb.init(project="MONAI_Spleen_3D_Segmentation", name="slice_image_exploration")
    # 🐝 log images to W&B
    log_spleen_slices(total_slices=100)
    # 🐝 finish the run
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Define Configuration

    Here, we define the configuration for dataloaders, models, train settings in a dictionary. Note that this config object would be passed to `wandb.init()` method to log all the necessary parameters that went into the experiment.
    """)
    return


@app.cell
def _(Norm):
    config = {
        # data
        "cache_rate": 1.0,
        "num_workers": 2,


        # train settings
        "train_batch_size": 2,
        "val_batch_size": 1,
        "learning_rate": 1e-3,
        "max_epochs": 100,
        "val_interval": 10, # check validation score after n epochs
        "lr_scheduler": "cosine_decay", # just to keep track




        # Unet model (you can even use nested dictionary and this will be handled by W&B automatically)
        "model_type": "unet", # just to keep track
        "model_params": dict(spatial_dims=3,
                      in_channels=1,
                      out_channels=2,
                      channels=(16, 32, 64, 128, 256),
                      strides=(2, 2, 2, 2),
                      num_res_units=2,
                      norm=Norm.BATCH,
        )
    }
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Define CacheDataset and DataLoader for training and validation

    Here we use `CacheDataset` to accelerate training and validation process, it's 10x faster than the regular Dataset.
    To achieve best performance, set `cache_rate=1.0` to cache all the data, if memory is not enough, set lower value.
    Users can also set `cache_num` instead of `cache_rate`, will use the minimum value of the 2 settings.
    And set `num_workers` to enable multi-threads during caching.
    If want to to try the regular Dataset, just change to use the commented code below.
    """)
    return


@app.cell
def _(
    CacheDataset,
    DataLoader,
    config,
    train_files,
    train_transforms,
    val_files,
    val_transforms,
):
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=config['cache_rate'], num_workers=config['num_workers'])
    # train_ds = Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=config['train_batch_size'], shuffle=True, num_workers=config['num_workers'])

    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=config['cache_rate'], num_workers=config['num_workers'])
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=config['val_batch_size'], num_workers=config['num_workers'])
    return train_ds, train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create Model, Loss, Optimizer and Scheduler
    """)
    return


@app.cell
def _(CosineAnnealingLR, DiceLoss, DiceMetric, UNet, config, torch):
    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(**config['model_params']).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epochs'], eta_min=1e-9)
    return device, dice_metric, loss_function, model, optimizer, scheduler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Execute a typical PyTorch training process
    """)
    return


@app.cell
def _(
    AsDiscrete,
    Compose,
    config,
    decollate_batch,
    device,
    dice_metric,
    loss_function,
    model,
    optimizer,
    os,
    root_dir,
    scheduler,
    sliding_window_inference,
    torch,
    train_ds,
    train_loader,
    val_loader,
    wandb,
):
    # 🐝 initialize a wandb run
    wandb.init(project='MONAI_Spleen_3D_Segmentation', config=config)
    wandb.watch(model, log_freq=100)
    max_epochs = config['max_epochs']
    val_interval = config['val_interval']
    best_metric = -1
    # 🐝 log gradients of the model to wandb
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    for epoch in range(max_epochs):
        print('-' * 10)
        print(f'epoch {epoch + 1}/{max_epochs}')
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (batch_data['image'].to(device), batch_data['label'].to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f'{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}')
            wandb.log({'train/loss': loss.item()})
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f'epoch {epoch + 1} average loss: {epoch_loss:.4f}')
        scheduler.step()
        wandb.log({'train/loss_epoch': epoch_loss})
        wandb.log({'learning_rate': scheduler.get_lr()[0]})
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for _val_data in val_loader:
                    val_inputs, val_labels = (_val_data['image'].to(device), _val_data['label'].to(device))
                    _roi_size = (160, 160, 160)  # 🐝 log train_loss for each step to wandb
                    _sw_batch_size = 4
                    _val_outputs = sliding_window_inference(val_inputs, _roi_size, _sw_batch_size, model)
                    _val_outputs = [post_pred(_i) for _i in decollate_batch(_val_outputs)]
                    val_labels = [post_label(_i) for _i in decollate_batch(val_labels)]
                    dice_metric(y_pred=_val_outputs, y=val_labels)
                metric = dice_metric.aggregate().item()
                wandb.log({'val/dice_metric': metric})  # step scheduler after each epoch (cosine decay)
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:  # 🐝 log train_loss averaged over epoch to wandb
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, 'best_metric_model.pth'))  # 🐝 log learning rate after each epoch to wandb
                    print('saved new best metric model')
                print(f'current epoch: {epoch + 1} current mean dice: {metric:.4f}\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}')
    print(f'\ntrain completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}')
    wandb.log({'best_dice_metric': best_metric, 'best_metric_epoch': best_metric_epoch})
    best_model_path = os.path.join(root_dir, 'best_metric_model.pth')
    model_artifact = wandb.Artifact('unet', type='model', description='Unet for 3D Segmentation of spleen', metadata=dict(config['model_params']))
    model_artifact.add_file(best_model_path)
    # 🐝 log best score and epoch number to wandb
    # 🐝 Version your model
    wandb.log_artifact(model_artifact)  # compute metric for current iteration  # 🐝 aggregate the final mean dice result  # 🐝 log validation dice score for each validation round  # reset the status for next validation round
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Check best model output with the input image and label
    """)
    return


@app.cell
def _(
    device,
    model,
    os,
    plt,
    root_dir,
    sliding_window_inference,
    torch,
    val_loader,
):
    model.load_state_dict(torch.load(os.path.join(root_dir, 'best_metric_model.pth')))
    model.eval()
    with torch.no_grad():
        for _i, _val_data in enumerate(val_loader):
            _roi_size = (160, 160, 160)
            _sw_batch_size = 4
            _val_outputs = sliding_window_inference(_val_data['image'].to(device), _roi_size, _sw_batch_size, model)
            plt.figure('check', (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f'image {_i}')
            plt.imshow(_val_data['image'][0, 0, :, :, 80], cmap='gray')  # plot the slice [:, :, 80]
            plt.subplot(1, 3, 2)
            plt.title(f'label {_i}')
            plt.imshow(_val_data['label'][0, 0, :, :, 80])
            plt.subplot(1, 3, 3)
            plt.title(f'output {_i}')
            plt.imshow(torch.argmax(_val_outputs, dim=1).detach().cpu()[0, :, :, 80])
            plt.show()
            if _i == 2:
                break
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log predictions to W&B in form of table
    """)
    return


@app.cell
def _(
    device,
    model,
    os,
    root_dir,
    sliding_window_inference,
    torch,
    val_loader,
    wandb,
):
    # 🐝 create a wandb table to log input image, ground_truth masks and predictions
    columns = ['filename', 'image', 'ground_truth', 'prediction']
    table = wandb.Table(columns=columns)
    model.load_state_dict(torch.load(os.path.join(root_dir, 'best_metric_model.pth')))
    model.eval()
    with torch.no_grad():
        for _i, _val_data in enumerate(val_loader):
            fn = _val_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]
            _roi_size = (160, 160, 160)
            _sw_batch_size = 4  # get the filename of the current image
            _val_outputs = sliding_window_inference(_val_data['image'].to(device), _roi_size, _sw_batch_size, model)
            for slice_no in range(80, 100):
                img = _val_data['image'][0, 0, :, :, slice_no]
                _label = _val_data['label'][0, 0, :, :, slice_no]
                prediction = torch.argmax(_val_outputs, dim=1).detach().cpu()[0, :, :, slice_no]
                table.add_data(fn, wandb.Image(img), wandb.Image(_label), wandb.Image(prediction))
    wandb.log({'val_predictions': table})
    # log predictions table to wandb with `val_predictions` as key
    # 🐝 Close your wandb run
    wandb.finish()  # log last 20 slices of each 3D image  # 🐝 Add data to wandb table dynamically    
    return


if __name__ == "__main__":
    app.run()
