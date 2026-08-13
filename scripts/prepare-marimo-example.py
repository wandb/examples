#!/usr/bin/env python3
"""Convert Jupyter notebooks into marimo examples with minimal diagnostics.

Usage:
  prepare-marimo-example.py notebook.ipynb --name example-name
  prepare-marimo-example.py notebook.ipynb --name example-name --force
  prepare-marimo-example.py notebook-path-list.txt --batch
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


class PrepareError(RuntimeError):
    """Expected user-facing input or setup error."""


MARIMO_DIR = Path("marimo")


def validate_repo_root(repo_root: Path) -> None:
    """Require the script to be run from the examples repository root."""

    if not (repo_root / MARIMO_DIR).is_dir():
        raise PrepareError(
            "Run this script from the root of the wandb/examples repository "
            "(the directory containing examples/marimo)."
        )


def resolve_file(raw_path: str, *, base_dir: Path) -> Path:
    """Resolve a file relative to one explicit base directory."""

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()

    if not path.is_file():
        raise PrepareError(f"input file does not exist: {raw_path}")
    return path


def display_path(path: Path, repo_root: Path) -> str:
    """Prefer repo-relative paths in messages and metadata."""

    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def slug_from_notebook(notebook: Path) -> str:
    """Derive a lowercase file-safe name from a notebook filename."""

    slug = re.sub(r"[^a-z0-9]+", "-", notebook.stem.lower()).strip("-")
    if not slug:
        raise PrepareError(f"could not derive a name from notebook: {notebook}")
    return slug


def validate_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise PrepareError(
            "--name must start with a letter or number and contain only "
            f"letters, numbers, '-' or '_': {name}"
        )


def run_command(argv: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a command and capture stdout/stderr as one diagnostic stream."""

    try:
        return subprocess.run(
            list(argv),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as error:
        return subprocess.CompletedProcess(
            args=list(argv),
            returncode=127,
            stdout=f"command not found: {argv[0]}\n{error}\n",
        )


def write_command_log(path: Path, result: subprocess.CompletedProcess[str]) -> None:
    """Write a self-contained transcript for a failed conversion attempt."""

    output = result.stdout or ""
    text = (
        f"$ {shlex.join(result.args)}\n\n"
        f"exit_code: {result.returncode}\n\n"
        "--- output ---\n\n"
        f"{output}"
    )
    if output and not output.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def write_diagnostics(
    *,
    debug_dir: Path,
    source: Path,
    target: Path,
    repo_root: Path,
    status: str,
    failed_stage: str | None,
    commands: dict[str, subprocess.CompletedProcess[str]],
) -> None:
    """Write result.json and, on failure, raw command transcripts."""

    # Successful reruns should not leave old failure logs behind.
    for old_log in debug_dir.glob("marimo-*.log"):
        old_log.unlink()

    command_metadata: dict[str, dict[str, object]] = {}
    for stage, result in commands.items():
        metadata: dict[str, object] = {
            "command": list(result.args),
            "exit_code": result.returncode,
        }

        if failed_stage is not None:
            log_path = debug_dir / f"marimo-{stage}.log"
            write_command_log(log_path, result)
            metadata["log"] = display_path(log_path, repo_root)

        command_metadata[stage] = metadata

    result = {
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "failed_stage": failed_stage,
        "source": display_path(source, repo_root),
        "target": display_path(target, repo_root),
        "target_exists": target.exists(),
        "commands": command_metadata,
    }
    (debug_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_notebook(
    source: Path,
    name: str,
    *,
    repo_root: Path,
    force: bool,
) -> str:
    """Convert one notebook, check it, and record the outcome."""

    if source.suffix.lower() != ".ipynb":
        raise PrepareError(f"input must be a .ipynb file: {source}")
    validate_name(name)

    target_dir = repo_root / MARIMO_DIR / "convert" / name
    target = target_dir / f"{name.replace('-', '_')}.py"
    debug_dir = target_dir / ".conversion"

    if target.exists() and not force:
        raise PrepareError(f"target already exists: {display_path(target, repo_root)}")

    target_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    commands: dict[str, subprocess.CompletedProcess[str]] = {}

    print(f"Converting {display_path(source, repo_root)}...", flush=True)
    commands["convert"] = run_command(
        ["uvx", "marimo", "convert", str(source), "-o", str(target)],
        cwd=repo_root,
    )

    if commands["convert"].returncode != 0:
        status = "conversion_failed"
        failed_stage = "convert"
    else:
        print(f"Checking {display_path(target, repo_root)}...", flush=True)
        commands["check"] = run_command(
            ["uvx", "marimo", "check", str(target)],
            cwd=repo_root,
        )
        failed_stage = "check" if commands["check"].returncode != 0 else None
        status = "check_failed" if failed_stage else "ok"

    write_diagnostics(
        debug_dir=debug_dir,
        source=source,
        target=target,
        repo_root=repo_root,
        status=status,
        failed_stage=failed_stage,
        commands=commands,
    )

    if failed_stage:
        log_path = debug_dir / f"marimo-{failed_stage}.log"
        print(f"{status.replace('_', ' ')}. See {display_path(log_path, repo_root)}", file=sys.stderr)
    else:
        print(f"Prepared {display_path(target, repo_root)}")

    return status


def iter_path_list(path_list: Path) -> list[str]:
    """Read non-empty, non-comment entries from a path-list file."""

    return [
        line.strip()
        for line in path_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def run_batch(
    path_list: Path,
    *,
    repo_root: Path,
    force: bool,
    fail_on_check: bool,
) -> int:
    """Prepare every notebook listed in a text file."""

    had_failure = False

    for raw_path in iter_path_list(path_list):
        try:
            # Batch entries are always relative to the repository root.
            source = resolve_file(raw_path, base_dir=repo_root)
            status = prepare_notebook(
                source,
                slug_from_notebook(source),
                repo_root=repo_root,
                force=force,
            )
        except PrepareError as error:
            print(error, file=sys.stderr)
            had_failure = True
            continue

        if status == "conversion_failed" or (status == "check_failed" and fail_on_check):
            had_failure = True

    return int(had_failure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Jupyter notebooks to marimo examples with minimal diagnostics."
    )
    parser.add_argument("input", help="A .ipynb notebook, or a path-list file with --batch.")
    parser.add_argument("--name", help="Target example name for a single notebook.")
    parser.add_argument("--batch", action="store_true", help="Treat input as a notebook path list.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing target.")
    parser.add_argument(
        "--fail-on-check",
        action="store_true",
        help="Exit non-zero when marimo check reports issues.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path.cwd().resolve()

    try:
        validate_repo_root(repo_root)

        if args.batch:
            if args.name:
                raise PrepareError("--name cannot be used with --batch")
            path_list = resolve_file(args.input, base_dir=repo_root)
            return run_batch(
                path_list,
                repo_root=repo_root,
                force=args.force,
                fail_on_check=args.fail_on_check,
            )

        if not args.name:
            raise PrepareError("--name is required unless --batch is used")

        source = resolve_file(args.input, base_dir=repo_root)
        status = prepare_notebook(
            source,
            args.name,
            repo_root=repo_root,
            force=args.force,
        )

        if status == "conversion_failed":
            return 1
        if status == "check_failed" and args.fail_on_check:
            return 1
        return 0

    except PrepareError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
