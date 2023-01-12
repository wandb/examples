FROM python:3.9-buster
ENV TERM xterm-256color
COPY "./requirements.txt" "requirements.txt"
RUN pip install -r requirements.txt
RUN mkdir /code
COPY ./deploy.py /code/deploy.py
WORKDIR /code

ENTRYPOINT ["python3", "deploy.py"]