marimo can export notebooks to several formats via the CLI.

```
> uvx marimo export --help
Usage: marimo export [OPTIONS]
                     COMMAND [ARGS]...

  Export a notebook to various formats.

Options:
  -h, --help  Show this message and exit.

Commands:
  html       Run a notebook and export it as an HTML file.
  html-wasm  Export a notebook as a WASM- powered marimo notebook. 
  ipynb      Export a marimo notebook as a Jupyter notebook
  md         Export a marimo notebook as a code fenced markdown file
  pdf        Export a marimo notebook as a PDF file.
  script     Export a marimo notebook as a flat script
  session    Execute a notebook or directory of notebooks and export session snapshots.
  thumbnail  Generate OpenGraph thumbnails for notebooks.
```

You can learn more about each option by calling the command with the `--help` flag. 

## PDF Export

Many people may be interested in exporting to a PDF. 

```bash
uvx marimo export pdf notebook.py -o notebook.pdf
```

PDF export uses `nbformat` and `nbconvert` under the hood. By default it uses the WebPDF exporter which requires Chromium. Install the dependencies:

```bash
uv pip install nbformat nbconvert
playwright install chromium
```

Useful flags:

- `--no-include-inputs` — hide code cells, show only outputs
- `--no-include-outputs` — include only code, skip outputs
- `--as=slides` — export as a slide deck PDF (uses reveal.js slide boundaries)
- `--raster-scale 4.0` — controls output sharpness (1.0–4.0, default 4.0)
- `--raster-server=live` — use when a widget needs a running Python kernel to render (recommended for slides)

## Script Export

```bash
uvx marimo export script notebook.py -o notebook.script.py
```

Flattens the notebook into a plain Python script in topological order.

## Common Flags

These flags work across most export subcommands:

- `-o`, `--output` — output file path
- `--watch` — re-export automatically when the notebook file changes
- `--sandbox` — run in an isolated `uv` environment
- `-f`, `--force` — overwrite if output file already exists
- `--` — pass CLI arguments to the notebook, e.g. `uvx marimo export html notebook.py -o out.html -- --arg value`
- `-y` automatic yes to prompts on the terminal `uvx marimo -y CMD ...` 
