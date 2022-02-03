# Simple fastai examples
A bunch of [fastai](https://github.com/fastai/fastai) examples using wandb through the `WandbCallback`.

Check [this](http://wandb.me/fastai_demo) report for a detailed wandb/fastai integration documentation.

# Install
You will need `fastai` and the `wandb` python packages. We really recommend `conda` for fastai installs as it will grab the latest fastai and pytorch and install cleanly in your conda environment:

```bash
conda install -c fastchan fastai
pip install wandb
```


# Python scripts
We propose 3 basic examples of fastai pipelines logging to wandb:

- [mnist.py](mnist.py): A simple classifier on top of MNIST, a `resnet18` backbone with default wandb logging through the `WandbCallback`. You can check a run [here](https://wandb.ai/tcapelle/fastai-mnist?workspace=user-tcapelle).

- [segmentation.py](segmentation.py): A simple Unet on top of CamVid-Tiny (a super tiny version of [Camvid](https://paperswithcode.com/dataset/camvid)). It also logs the dataset and the best model checkpoint at the end. You can visualize some of the predictions [here](https://wandb.ai/tcapelle/fastai-camvid/runs/3nfmevj0?workspace=user-tcapelle) and the [model checkpoint](https://wandb.ai/tcapelle/fastai-camvid/artifacts/model/run-3nfmevj0-model/66ed3380c1cbc0769c8d/files).

- [tabular.py](fastai-tabular.py): A tabular/dataframe based example, training a fastai [TabularModel](https://docs.fast.ai/tutorial.tabular.html). A run can be found [here](https://wandb.ai/tcapelle/fastai-tabular/runs/dfmyy7vz?workspace=user-tcapelle).

