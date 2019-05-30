# Finding The Witness Puzzle Patterns

This repo is example code and data for training a CNN to identify patterns from the video game "The Witness".

To train the model:

```sh
# Install requirements.
pip install -r requirements.txt

# Login to W&B -- this will save your run results.
wandb login

# Process data from /data/all into a training set in /data/train and validation set in /data/valid.
./process.py

# Train your model, and save the best one into the /model folder!
./train.py
```

To visualize output:

```sh
# Run a prediction on all entries in your validation set.
./predict.py

# Generate a visualization of every layer in the model.
./visualize.py
```
