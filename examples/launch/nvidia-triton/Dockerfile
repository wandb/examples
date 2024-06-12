from nvcr.io/nvidia/tritonserver:22.11-py3

RUN pip install wandb
RUN mkdir /code
RUN mkdir /model_repository
COPY ./deploy.py  /code/deploy.py

WORKDIR /code

ENTRYPOINT ["python3", "deploy.py"]
