FROM tensorflow/tensorflow:latest
COPY . /src
WORKDIR /src
RUN pip install wandb pillow
ENTRYPOINT [ "env", "python3", "train.py"]