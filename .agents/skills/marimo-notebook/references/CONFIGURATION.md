# Configuration

## Two Scopes

1. **App config** — per-notebook, stored in the `.py` file header. Configure via the gear icon (top-right): notebook width, title, custom CSS, custom HTML head.
2. **User config** — global, typically stored in `~/.config/marimo/marimo.toml`. Runtime, display, hotkeys, autosave, formatting, server settings.

## Priority (Highest → Lowest)

1. PEP 723 script metadata block in the notebook file
2. `pyproject.toml` — project-level overrides
3. User config (`marimo.toml`) — global defaults

## pyproject.toml

```toml
[tool.marimo.formatting]
line_length = 120

[tool.marimo.display]
default_width = "full"

[tool.marimo.runtime]
default_sql_output = "native"
watcher_on_save = "autorun"
```

## Config Discovery

marimo searches for `.marimo.toml` in: current directory → parent directories → home directory → XDG config directory.

## Useful Commands

```bash
marimo config show       # view current config and file location
marimo config describe   # list all available config options
```
