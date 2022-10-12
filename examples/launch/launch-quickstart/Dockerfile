FROM python:3.8
COPY . /src
WORKDIR /src
RUN pip install wandb torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
ENTRYPOINT [ "env", "python3", "train.py"]