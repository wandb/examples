# Agent skills

Skills for AI coding agents working in this repository, in the generic
[Agent Skills](https://agentskills.io) format: each skill is a directory
containing a `SKILL.md` with YAML frontmatter (`name`, `description`),
plus optional `references/` files.

## Skills

| Skill | Purpose |
| --- | --- |
| [`marimo-example-notebook`](marimo-example-notebook/SKILL.md) | **Start here** for creating or refactoring example notebooks in this repo. Encodes wandb/examples conventions and best practices. |
| [`marimo-notebook`](marimo-notebook/SKILL.md) | General marimo notebook format and mechanics (vendored). |
| [`jupyter-to-marimo`](jupyter-to-marimo/SKILL.md) | Converting existing Jupyter notebooks to marimo (vendored). |

## Vendored skills

`marimo-notebook` and `jupyter-to-marimo` are vendored verbatim from
[marimo-team/skills](https://github.com/marimo-team/skills) (Apache-2.0,
LICENSE included in each directory) so agents can use them without
network access.

- Upstream commit: `62d78d97278e0517c2270a8fbafd3f95a59df9cd`
- Vendored: 2026-07-20

To refresh, re-copy `marimo-notebook/` and `jupyter-to-marimo/`
from upstream and update the commit SHA above. Do not edit vendored files
in place — repo-specific guidance belongs in `marimo-example-notebook`.
