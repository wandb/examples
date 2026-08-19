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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-model-registry/Model_Registry_E2E.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{model-reg-e2e} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{model-reg-e2e} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Registry Tutorial
    The model registry is a central place to house and organize all the model tasks and their associated artifacts being worked on across an org:
    - Model checkpoint management
    - Document your models with rich model cards
    - Maintain a history of all the models being used/deployed
    - Facilitate clean hand-offs and stage management of models
    - Tag and organize various model tasks
    - Set up automatic notifications when models progress

    This tutorial will walkthrough how to track the model development lifecycle for a simple image classification task.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🛠️ Install `wandb`
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install -q wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Login to W&B
    - You can explicitly login using `wandb login` or `wandb.login()` (See below)
    - Alternatively you can set environment variables. There are several env variables which you can set to change the behavior of W&B logging. The most important are:
        - `WANDB_API_KEY` - create a new API key in your "Settings" section under your profile at [wandb.ai/settings](https://wandb.ai/settings)
        - `WANDB_BASE_URL` - this is the url of the W&B server
    - Create a new API key in "Profile" -> "Settings" in the W&B App. Store your API key securely. It can only be viewed once when created.

    ![api_token](https://drive.google.com/uc?export=view&id=1Xn7hnn0rfPu_EW0A_-32oCXqDmpA0-kx)
    """)
    return


@app.cell
def _():
    import wandb

    # Login to W&B
    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Log Data and Model Checkpoints as Artifacts
    W&B Artifacts allows you to track and version arbitrary serialized data (e.g. datasets, model checkpoints, evaluation results). When you create an artifact, you give it a name and a type, and that artifact is forever linked to the experimental system of record. If the underlying data changes, and you log that data asset again, W&B will automatically create new versions through checksummming its contents. W&B Artifacts can be thought of as a lightweight abstraction layer on top of shared unstructured file systems.

    ### Anatomy of an artifact

    The `Artifact` class will correspond to an entry in the W&B Artifact registry.  The artifact has
    * a name
    * a type
    * metadata
    * description
    * files, directory of files, or references

    Example usage:
    ```
    run = wandb.init(project = "my-project")
    artifact = wandb.Artifact(name = "my_artifact", type = "data")
    artifact.add_file("/path/to/my/file.txt")
    run.log_artifact(artifact)
    run.finish()
    ```

    In this tutorial, the first thing we will do is download a training dataset and log it as an artifact to be used downstream in the training job.
    """)
    return


@app.cell
def _(wandb):
    import sys
    from pathlib import Path

    # FORM VARIABLES
    PROJECT_NAME = "model-registry-tutorial"
    ENTITY = wandb.api.default_entity # replace with your Team name or username

    # Dataset constants
    DATASET_NAME = "nature_100"
    DATA_DIR = (Path(sys.path[0]) / "data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_SRC = DATA_DIR / DATASET_NAME
    IMAGES_PER_LABEL = 10
    BALANCED_SPLITS = {"train" : 8, "val" : 1, "test": 1}
    MODEL_TYPE = "squeezenet"
    return DATASET_NAME, DATA_DIR, DATA_SRC, ENTITY, PROJECT_NAME, Path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's grab a version of our Dataset
    """)
    return


@app.cell
def _(DATASET_NAME, DATA_DIR):
    import requests, zipfile, io

    # Download the dataset from a bucket
    src_url = f"https://storage.googleapis.com/wandb_datasets/{DATASET_NAME}.zip"
    src_zip = f"{DATASET_NAME}.zip"

    # Download the zip file from W&B
    r = requests.get(src_url)

    # Create a file object using the string data
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=DATA_DIR)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are going to generate a file containing the image
    """)
    return


@app.cell
def _(DATASET_NAME, DATA_SRC, ENTITY, PROJECT_NAME, wandb):
    with wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type='log_datasets') as run:
      train_art = wandb.Artifact(name=DATASET_NAME, 
                                 type='raw_images', 
                                 description='nature image dataset with 10 classes, 10 images per class')
      train_art.add_dir(DATA_SRC)
      wandb.log_artifact(train_art)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using Artifact names and aliases to easily hand-off and abstract data assets
    - By simply referring to the `name:alias` combination of a dataset or model, we can better standardize components of a workflow
    - For instance, you can build PyTorch `Dataset`'s or `DataModule`'s which take as arguments W&B Artifact names and aliases to load appropriately

    You can now see all the metadata associated with this dataset, the W&B runs consuming it, and the whole lineage of upstream and downstream artifacts!

    ![api_token](https://drive.google.com/uc?export=view&id=1fEEddXMkabgcgusja0g8zMz8whlP2Y5P)
    """)
    return


@app.cell
def _(DATA_SRC, Path, wandb):
    import math

    from PIL import Image
    import torch
    from torchvision import transforms, models
    from torch.utils.data import Dataset, DataLoader, random_split

    class NatureDataset(Dataset):
        def __init__(self, artifact_name_alias: str, transform=None):
            self.transform = transform

            # Pull down the artifact locally to load it into memory
            art = wandb.use_artifact(artifact_name_alias)
            self.path_at = Path(art.download())

            self.img_paths = list(DATA_SRC.rglob("*.jpg"))
            labels = [image_path.parent.name for image_path in self.img_paths]
            self.class_names = sorted(set(labels))
            self.idx_to_class = {k: v for k, v in enumerate(self.class_names)}
            self.class_to_idx = {v: k for k, v in enumerate(self.class_names)}

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            image_path = Path(self.path_at) / self.img_paths[idx]

            image = Image.open(image_path)
            label = image_path.parent.name
            label = torch.tensor(self.class_to_idx[label], dtype=torch.long)

            if self.transform:
                image = self.transform(image)

            return image, label
    
    class Dataloaders:
        def __init__(self,
                     artifact_name_alias: str,
                     batch_size: int,
                     input_size: int,
                     seed: int = 42):
            self.artifact_name_alias = artifact_name_alias
            self.batch_size = batch_size
            self.input_size = input_size
            self.seed = seed

            tfms = transforms.Compose([transforms.ToTensor(),
                                       transforms.CenterCrop(self.input_size),
                                       transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))])

            print(f"Setting up data from artifact: {self.artifact_name_alias}")
            self.dataset = NatureDataset(artifact_name_alias=self.artifact_name_alias,
                                         transform=tfms)

            nature_length = len(self.dataset)
            train_size = math.floor(0.8 * nature_length)
            val_size = math.floor(0.2 * nature_length)
            print(f"Splitting dataset into {train_size} training samples and {val_size} validation samples")
            self.ds_train, self.ds_valid = random_split(
                self.dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed))
    
            self.train = DataLoader(self.ds_train, batch_size=self.batch_size)
            self.valid = DataLoader(self.ds_valid, batch_size=self.batch_size)

    return models, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Writing the Model Class and Validation Function
    """)
    return


@app.cell
def _(models, torch, wandb):
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(num_classes, feature_extract, use_pretrained=True):
        "Create a model from torchvision.models"
        model_ft = None

        # SqueezeNet
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes

        return model_ft, 224

    class NaturePyTorchModule(torch.nn.Module):
        def __init__(self,
                     model_name,
                     num_classes=10,
                     feature_extract=True,
                     lr=0.01):
            '''method used to define our model parameters'''
            super().__init__()

            self.model_name = model_name
            self.num_classes = num_classes
            self.feature_extract = feature_extract
            self.lr = lr
            self.model, self.input_size = initialize_model(num_classes=self.num_classes,
                                                           feature_extract=True)

        def forward(self, x):
            '''method used for inference input -> output'''
            return self.model(x)

    def evaluate_model(model, val_dl, idx_to_class, class_names):
        device = torch.device("cpu")
        model.eval()
        test_loss = 0
        correct = 0
        preds = []
        actual = []
    
        val_table = wandb.Table(columns=['pred', 'actual', 'image'])

        with torch.no_grad():
            for data, target in val_dl:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                preds += list(pred.flatten().tolist())
                actual += target.numpy().tolist()
                correct += pred.eq(target.view_as(pred)).sum().item()

                for idx, img in enumerate(data):
                    img = img.numpy().transpose(1, 2, 0)
                    pred_class = idx_to_class[pred.numpy()[idx][0]]
                    target_class = idx_to_class[target.numpy()[idx]]
                    val_table.add_data(pred_class, target_class, wandb.Image(img))

        test_loss /= len(val_dl.dataset)
        accuracy = 100.0 * correct / len(val_dl.dataset)
        conf_mat = wandb.plot.confusion_matrix(y_true=actual, preds=preds, class_names=class_names)
        return test_loss, accuracy, preds, val_table, conf_mat

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tracking the Training Loop
    During training, it is a best practice to checkpoint your models overtime, so if training gets interrupted or your instance crashes you can resume from where you left off. With artifact logging, we can track all our checkpoints with W&B and attach any metadata we want (like format of serialization, class labels, etc.). That way, when someone needs to consume a checkpoint they know how to use it. When logging models of any form as artifacts, ensure to set the `type` of the artifact to `model`.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%wandb -h 600
    # 
    # run = wandb.init(project=PROJECT_NAME,
    #                      entity=ENTITY,
    #                      job_type='training',
    #                      config={'model_type': MODEL_TYPE,
    #                              'lr': 1.0,
    #                              'gamma': 0.75,
    #                              'batch_size': 16,
    #                              'epochs': 5})
    # 
    # model = NaturePyTorchModule(wandb.config['model_type'])
    # 
    # wandb.config['input_size'] = 224
    # 
    # dls = Dataloaders(artifact_name_alias=f"{DATASET_NAME}:latest",
    #                   batch_size=wandb.config['batch_size'],
    #                   input_size=wandb.config['input_size'])
    # 
    # # Train the model
    # learning_rate = wandb.config["lr"]
    # gamma = wandb.config["gamma"]
    # epochs = wandb.config["epochs"]
    # 
    # device = torch.device("cpu")
    # optimizer = optim.Adadelta(model.parameters(), lr=wandb.config['lr'])
    # scheduler = StepLR(optimizer, step_size=1, gamma=wandb.config['gamma'])
    # 
    # best_loss = float("inf")
    # best_model = None
    # 
    # for epoch_ndx in range(epochs):
    #     model.train()
    #     for batch_ndx, batch in enumerate(dls.train):
    #         data, target = batch[0].to("cpu"), batch[1].to("cpu")
    #         optimizer.zero_grad()
    #         preds = model(data)
    #         loss = F.cross_entropy(preds, target)
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    # 
    #         ### Log your metrics ###
    #         wandb.log({
    #             "train/epoch_ndx": epoch_ndx,
    #             "train/batch_ndx": batch_ndx,
    #             "train/train_loss": loss,
    #             "train/learning_rate": optimizer.param_groups[0]["lr"]
    #         })
    #         print(f"Epoch: {epoch_ndx}, Batch: {batch_ndx}, Loss: {loss}")
    # 
    #     ### Evaluation at the end of each epoch ###
    #     test_loss, accuracy, preds, val_table, conf_mat = evaluate_model(
    #         model,
    #         dls.valid, 
    #         dls.dataset.idx_to_class,
    #         dls.dataset.class_names,
    #     )
    # 
    #     is_best = test_loss < best_loss
    # 
    #     wandb.log({
    #         'eval/test_loss': test_loss,
    #         'eval/accuracy': accuracy,
    #         'eval/conf_mat': conf_mat,
    #         'eval/val_table': val_table})
    # 
    #   ### Checkpoing your model weights ###
    #     torch.save(model.state_dict(), "model.pth")
    #     art = wandb.Artifact(f"nature-{wandb.run.id}", 
    #                         type="model",
    #                         metadata={'format': 'onnx',
    #                                   'num_classes': len(dls.dataset.class_names),
    #                                   'model_type': wandb.config['model_type'],
    #                                   'model_input_size': wandb.config['input_size'],
    #                                   'index_to_class': dls.dataset.idx_to_class})
    # 
    #     art.add_file("model.pth")
    # 
    #     ### Add aliases to keep track of your best checkpoints over time
    #     wandb.log_artifact(art, aliases=["best", "latest"] if is_best else None)
    #     if is_best:
    #         best_model = art
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Manage all your model checkpoints for a project under one roof.

    ![api_token](https://drive.google.com/uc?export=view&id=1z7nXRgqHTPYjfR1SoP-CkezyxklbAZlM)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Registry
    After logging a bunch of checkpoints across multiple runs during experimentation, now comes time to hand-off the best checkpoint to the next stage of the workflow (e.g. testing, deployment).

    The Model Registry is a central page that lives above individual W&B projects. It houses **Registered Models**, portfolios that store "links" to the valuable checkpoints living in individual W&B Projects.

    The model registry offers a centralized place to house the best checkpoints for all your model tasks. Any `model` artifact you log can be "linked" to a Registered Model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Creating **Registered Models** and Linking through the UI
    #### 1. Access your team's model registry by going the team page and selecting `Model Registry`

    ![model registry](https://drive.google.com/uc?export=view&id=1ZtJwBsFWPTm4Sg5w8vHhRpvDSeQPwsKw)

    #### 2. Create a new Registered Model.

    ![model registry](https://drive.google.com/uc?export=view&id=1RuayTZHNE0LJCxt1t0l6-2zjwiV4aDXe)

    #### 3. Go to the artifacts tab of the project that holds all your model checkpoints

    ![model registry](https://drive.google.com/uc?export=view&id=1LfTLrRNpBBPaUb_RmBIE7fWFMG0h3e0E)

    #### 4. Click "Link to Registry" for the model artifact version you want.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Creating Registered Models and Linking through the **API**
    You can [link a model via api](https://docs.wandb.ai/guides/models) with `wandb.run.link_artifact` passing in the artifact object, and the name of the **Registered Model**, along with aliases you want to append to it. **Registered Models** are entity (team) scoped in W&B so only members of a team can see and access the **Registered Models** there. You indicate a registered model name via api with `<entity>/model-registry/<registered-model-name>`. If a Registered Model doesn't exist, one will be created automatically.
    """)
    return


@app.cell
def _(ENTITY, best_model, wandb):
    if ENTITY is not None:
      wandb.run.link_artifact(best_model, f'{ENTITY}/model-registry/Model Registry Tutorial', aliases=['staging'])
    else:
      print('Must indicate entity where Registered Model will exist')
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What is "Linking"?
    When you link to the registry, this creates a new version of that Registered Model, which is just a pointer to the artifact version living in that project. There's a reason W&B segregates the versioning of artifacts in a project from the versioning of a Registered Model. The process of linking a model artifact version is equivalent to "bookmarking" that artifact version under a Registered Model task.

    Typically during R&D/experimentation, researchers generate 100s, if not 1000s of model checkpoint artifacts, but only one or two of them actually "see the light of day." This process of linking those checkpoints to a separate, versioned registry helps delineate the model development side from the model deployment/consumption side of the workflow. The globally understood version/alias of a model should be unpolluted from all the experimental versions being generated in R&D and thus the versioning of a Registered Model increments according to new "bookmarked" models as opposed to model checkpoint logging.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create a Centralized Hub for all your models
    - Add a model card, tags, slack notifactions to your Registered Model
    - Change aliases to reflect when models move through different phases
    - Embed the model registry in reports for model documentation and regression reports. See this report as an [example](https://api.wandb.ai/links/wandb-smle/r82bj9at)
    ![model registry](https://drive.google.com/uc?export=view&id=1lKPgaw-Ak4WK_91aBMcLvUMJL6pDQpgO)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Set up Slack Notifications when new models get linked to the registry

    ![model registry](https://drive.google.com/uc?export=view&id=1RsWCa6maJYD5y34gQ0nwWiKSWUCqcjT9)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Consuming a Registered Model
    You now can consume any registered model via API by referring the corresponding `name:alias`. Model consumers, whether they are engineers, researchers, or CI/CD processes, can go to the model registry as the central hub for all models that should "see the light of day": those that need to go through testing or move to production.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%wandb -h 600
    # 
    # run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type='inference')
    # artifact = run.use_artifact(f'{ENTITY}/model-registry/Model Registry Tutorial:staging', type='model')
    # artifact_dir = artifact.download()
    # wandb.finish()
    return


if __name__ == "__main__":
    app.run()
