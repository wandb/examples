FROM python:3.11

RUN pip install wandb==0.15.5 transformers==4.27.1 datasets torch

COPY train.py .

ENTRYPOINT ["python", "train.py"]
