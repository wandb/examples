#!/bin/bash
sudo apt-get update
sudo apt-get install python3-venv
python3 -m venv jx
source jx/bin/activate
pip install --upgrade pip
mkdir projects
cd projects
pip install requests
git clone https://github.com/kldarek/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
git checkout -b proto
pip install datasets
pip install ".[flax,testing,sentencepiece]"
pip install -r examples/flax/_tests_requirements.txt
pip install tensorflow
pip install wandb
pip install -U tensorboard-plugin-profile
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
curl -fsSL https://code-server.dev/install.sh | sh