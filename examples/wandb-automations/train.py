import random, wandb

PROJECT = "another_launch_project"
# ENTITY = "wandb"
ENTITY = "launch-test"
# ENTITY = None

default_config = dict(lr=0.001,epochs=10)

def train(config):
    run = wandb.init(project=PROJECT, entity=ENTITY, config=config)
    run.log_code()
    config = wandb.config
    for epoch in range(config.epochs):
        run.log({"loss": random.random(), "epoch": epoch})
    run.finish()

if __name__ == "__main__": train(default_config)
    
