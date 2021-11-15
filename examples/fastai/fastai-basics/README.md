# Simple core fastai examples

We propose 3 basic examples of fastai pipelines logging to wandb:

- [fastai-mnist.py](fastai-mnist.py): A simple classifier on top of MNIST, a `resnet18` backbone with default wandb logging though the `WandbCallback`. You can check a run [here](https://wandb.ai/tcapelle/fastai-mnist?workspace=user-tcapelle)

- [fastai-segmentation.py](fastai-segmentation.py): A simple Unet on top of CamVid-Tiny (a super tiny version of [Camvid](https://paperswithcode.com/dataset/camvid). It also logs the dataset and the model checkpoint at the end. You can visualize some of the predictions [here](https://wandb.ai/tcapelle/fastai-camvid/runs/3nfmevj0?workspace=user-tcapelle) and the [model checkpoint](https://wandb.ai/tcapelle/fastai-camvid/artifacts/model/run-3nfmevj0-model/66ed3380c1cbc0769c8d/files)

- [fastai-tabular.py](fastai-tabular.py): A tabular/dataframe based example, training a fastai [TabularModel](https://docs.fast.ai/tutorial.tabular.html). A run can be found [here](https://wandb.ai/tcapelle/fastai-tabular/runs/dfmyy7vz?workspace=user-tcapelle)