#!/bin/usr/python

import os
import argparse

# no_longer = {
#         "RayTune_with_wandb": "",
#         "Weights_&_Biases_with_fastai": "",
#         "WandB_Prompts_Quickstart":"",    
# }

title_mapping = {
    "Intro_to_Weights_&_Biases": "experiments",
    "Pipeline_Versioning_with_W&B_Artifacts": "artifacts",
    "Model_Registry_E2E": "models",
    "W&B_Tables_Quickstart": "tables",
    "Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B": "sweeps",
    "Using_W&B_Sweeps_with_XGBoost": "xgboost_sweeps",
    "Simple_PyTorch_Integration": "pytorch",
    "Huggingface_wandb": "huggingface",
    "Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps": "tensorflow_sweeps",
    "Image_Classification_using_PyTorch_Lightning": "lightning",
    "Simple_TensorFlow_Integration": "tensorflow",
    "Use_WandbMetricLogger_in_your_Keras_workflow": "keras",
    "Use_WandbEvalCallback_in_your_Keras_workflow": "keras_table",
    "Use_WandbModelCheckpoint_in_your_Keras_workflow": "keras_models",
}

def rename_markdown_file(filename, title_names):
    "Checking if we need to rename markdown file..."
    # Check if .ipynb name exists in our mapping
    base_name = os.path.basename(filename).split('.')[0]
    if base_name in title_names:
        new_filename = title_names[base_name]

        # Rename file
        print(f"Renaming notebook from {filename} to {new_filename}.md")
        os.rename(filename, new_filename+".md")
    else:
        print(f"No title match found. {filename} reserved.")


def main(args):
    print(args.file)
    for markdown_file in args.file:
        rename_markdown_file(markdown_file, title_mapping)
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="*", help="Notebook to check if it needs converting")
    args = parser.parse_args()
    main(args)