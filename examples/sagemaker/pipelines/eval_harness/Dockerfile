FROM winglian/axolotl:main-py3.11-cu121-2.1.2

WORKDIR /workspace/eval_harness

RUN git config --global --add safe.directory /workspace/eval_harness

# Install Eval harness
RUN python -m pip install "lm_eval[wandb] @ git+https://github.com/ayulockin/lm-evaluation-harness@wandb-logging"

# install other requirements
COPY . /workspace/eval_harness

RUN python -m pip install -r requirements.txt

ENTRYPOINT ["python", "run.py"]