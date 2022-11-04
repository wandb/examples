FROM python:3.9-slim-bullseye
LABEL maintainer='Weights & Biases <support@wandb.com>'
LABEL version="0.1"

# Needed to compile wandb dependency psutil
RUN apt-get update -y && apt-get install -y gcc
RUN pip install pandas sklearn seaborn tqdm wandb

WORKDIR /app
COPY train* .
ENTRYPOINT ["python", "train.py"]
