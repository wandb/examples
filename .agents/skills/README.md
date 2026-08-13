# Agent skills

Skills for AI coding agents working in this repository, in the generic
[Agent Skills](https://agentskills.io) format: each skill is a directory
containing a `SKILL.md` with YAML frontmatter (`name`, `description`),
plus optional `references/` files.

## Skills

| Skill | Purpose |
| --- | --- |
| [`marimo-wandb-notebooks`](marimo-wandb-notebooks/SKILL.md) | **Start here** for creating or refactoring example notebooks in this repo. Encodes wandb/examples conventions and best practices. |

## Scripts

Use [`../../scripts/prepare-marimo-example.py`](../../scripts/prepare-marimo-example.py)
to create the initial marimo notebook from a Jupyter `.ipynb`, capture
`marimo check` output, and write a temporary `.conversion/` report directory
for the polishing pass. The report directory includes a Markdown handoff,
raw convert/check output, structured JSON metadata, and an event log.
