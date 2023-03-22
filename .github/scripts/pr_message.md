Thanks for contributing to `wandb/examples`!
We appreciate your efforts in opening a PR for the examples repository. Our goal is to ensure a smooth and enjoyable experience for you 😎.

## Guidelines
The examples repo is regularly tested against the ever-evolving ML stack. To facilitate our work, please adhere to the following guidelines:

- Notebook naming: You can use a combination of snake_case and CamelCase for your notebook name. Avoid using spaces (replace them with `_`) and special characters 
(`&%$?`). For example:
```
Cool_Keras_integration_example_with_weights_and_biases.ipynb 
```
is acceptable, but
```
Cool Keras Example with W&B.ipynb
```
is not. Avoid spaces and the `&` character. To refer to W&B, you can use: `weights_and_biases` or just `wandb` (it's our library, after all!)

- Managing dependencies within the notebook: You may need to set up dependencies to ensure that your code works. Please avoid the following practices:
    - Docker-related activities. If Docker installation is required, consider adding a full example with the corresponding `Dockerfile` to the `wandb/examples/examples` 
folder (where non-Colab examples reside).
    - Using `pip install` as the primary method to install packages. When calling `pip` in a cell, avoid performing other tasks. We automatically filter these types of 
cells, and executing other actions might break the automatic testing of the notebooks. For example, 
    ```
    pip install -qU wandb transformers gpt4
    ```
    is acceptable, but
    ```python
    pip install -qU wandb
    import wandb
    ```
    is not.
    - Installing packages from a GitHub branch. Although it's acceptable 😎 to directly obtain the latest bleeding-edge libraries from `GitHub`, did you know that you can 
install them like this:

    ```bash
    !pip install -q git+https://github.com/huggingface/transformers
    ```
    > You don't need to clone, then `cd` into the repo and install it in editable mode.
    
    - Avoid referencing specific Colab directories. Google Colab has a `/content` directory where everything resides. Avoid explicitly referencing this directory because 
we test our notebooks with pure Jupyter (without Colab). Instead, use relative paths to make the notebook reproducible.

- The Jupyter notebook file `.ipynb` is nothing more than a JSON file with primarily two types of cells: markdown and code. There is also a bunch of other metadata 
specific to Google Colab. We have a set of tools to ensure proper notebook formatting. These tools can be found at [wandb/nb_helpers](https://github.com/wandb/nb_helpers).

> Before merging, wait for a maintainer to `clean` and format the notebooks you're adding. You can tag @tcapelle.

### Before marking the PR as ready for review, please run your notebook one more time. Restart the Colab and run all. We will provide you with links to open the Colabs below