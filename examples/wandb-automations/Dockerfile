FROM pytorch/pytorch:latest

RUN python3 -m pip  install "timm==0.6.12" wandb tqdm

COPY . /app

CMD ["python3", "/app/eval_fmnist.py"]
