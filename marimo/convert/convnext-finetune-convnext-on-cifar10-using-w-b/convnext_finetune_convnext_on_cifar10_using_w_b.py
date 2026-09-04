# /// script
# dependencies = ["https://download-pytorch-org/whl/torch-stable-html", "six", "tensorboardx", "timm", "torch", "torchvision", "wandb"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/convnext/Finetune_ConvNext_on_CIFAR10_using_W&B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use this notebook to finetune a ConvNeXt-tiny model on CIFAR 10 dataset. The [official ConvNeXt repository](https://github.com/facebookresearch/ConvNeXt) is instrumented with [Weights and Biases](https://wandb.ai/site). You can now easily log your train/test metrics and version control your model checkpoints to Weigths and Biases
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ⚽️ Installation and Setup

    The following installation instruction is based on [INSTALL.md](https://github.com/facebookresearch/ConvNeXt/blob/main/INSTALL.md) provided by the official ConvNeXt repository.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: torch==1.8.0+cu111 torchvision==0.9.0+cu111 https://download.pytorch.org/whl/torch_stable.html !pip install -qq torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    # packages added via marimo's package management: wandb timm==0.3.2 six tensorboardX !pip install -qq wandb timm==0.3.2 six tensorboardX
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Download the official ConvNeXt respository.
    """)
    return


@app.cell
def _(subprocess):
    #! git clone --depth 1 https://github.com/facebookresearch/ConvNeXt
    subprocess.call(['git', 'clone', '--depth', '1', 'https://github.com/facebookresearch/ConvNeXt'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🏀 Download the Dataset

    We will be finetuning on CIFAR-10 dataset. To use any custom dataset (CIFAR-10 here) the format of the dataset should be as shown below:

    ```
    /path/to/dataset/
      train/
        class1/
          img1.jpeg
        class2/
          img2.jpeg
      val/
        class1/
          img3.jpeg
        class2/
          img4.jpeg
    ```

    I have used this [repository](https://github.com/YoongiKim/CIFAR-10-images) that has the CIFAR-10 images in the required format.
    """)
    return


@app.cell
def _(subprocess):
    #! git clone --depth 1 https://github.com/YoongiKim/CIFAR-10-images
    subprocess.call(['git', 'clone', '--depth', '1', 'https://github.com/YoongiKim/CIFAR-10-images'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🏈 Download Pretrained Weights

    We will be finetuning the ConvNeXt Tiny model pretrained on ImageNet 1K dataset.
    """)
    return


@app.cell
def _(subprocess):
    import os
    os.chdir('ConvNeXt/')
    #! wget https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    subprocess.call(['wget', 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🎾 Train with Weights and Biases

    If you want to log the train and evaluation metrics using Weights and Biases pass `--enable_wandb true`.

    You can also save the finetuned checkpoints as version controlled W&B [Artifacts](https://docs.wandb.ai/guides/artifacts) if you pass `--wandb_ckpt true`.
    """)
    return


@app.cell
def _(subprocess):
    #! python main.py --epochs 10 --model convnext_tiny --data_set image_folder --data_path ../CIFAR-10-images/train --eval_data_path ../CIFAR-10-images/test --nb_classes 10 --num_workers 8 --warmup_epochs 0 --save_ckpt true --output_dir model_ckpt --finetune convnext_tiny_1k_224_ema.pth --cutmix 0 --mixup 0 --lr 4e-4 --enable_wandb true --wandb_ckpt true
    subprocess.call(['python', 'main.py', '--epochs', '10', '--model', 'convnext_tiny', '--data_set', 'image_folder', '--data_path', '../CIFAR-10-images/train', '--eval_data_path', '../CIFAR-10-images/test', '--nb_classes', '10', '--num_workers', '8', '--warmup_epochs', '0', '--save_ckpt', 'true', '--output_dir', 'model_ckpt', '--finetune', 'convnext_tiny_1k_224_ema.pth', '--cutmix', '0', '--mixup', '0', '--lr', '4e-4', '--enable_wandb', 'true', '--wandb_ckpt', 'true'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🏐 Conclusion

    * **The above setting gives a top-1 accuracy of ~95%.**
    * The ConvNeXt repository comes with modern training regimes and is easy to finetune on any dataset.
    * The finetune model achieves competitive results.

    * By passing two arguments you get the following:

      * Repository of all your experiments (train and test metrics) as a [W&B Project](https://docs.wandb.ai/ref/app/pages/project-page). You can easily compare experiments to find the best performing model.
      * Hyperparameters (Configs) used to train individual models.
      * System (CPU/GPU/Disk) metrics.
      * Model checkpoints saved as W&B Artifacts. They are versioned and easy to share.

      Check out the associated [W&B run page](https://wandb.ai/ayut/convnext/runs/16vi9e31). $→$
    """)
    return


if __name__ == "__main__":
    app.run()
