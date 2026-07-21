## Running notebooks

You can deploy a single marimo notebook as a web app:

```bash
uvx marimo run --sandbox notebook.py
```

The `--sandbox` flag makes sure the notebook runs in an isolated UV environment.

Or deploy a folder of notebooks as a web app with multiple notebooks. Also here, you can use the `--sandbox` flag to run each notebook in its own isolated environment, using the PEP 723 dependencies declared in each notebook:

```bash
uvx marimo run --sandbox <folder>
```

### Thumbnails

When you host multiple notebooks you may want to generate thumbnails. You can generate OpenGraph thumbnails for notebooks using:

```bash
uvx marimo export thumbnail notebook.py
uvx marimo export thumbnail folder/
```

Thumbnails are stored at `__marimo__/assets/<notebook_stem>/opengraph.png`. The user may also put screenshots there manually. 

Besides images, you can also add metadata to the notebooks by adding to the PEP 723 Dependencies on top of the file. These will appear in an overview if the user deploys a folder of notebooks. 

```
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars==1.37.1",
# ]
# [tool.marimo.opengraph]
# title = "My dashboard"
# description = "Tracking my portfolio over time"
# ///
```
