#!/usr/bin/env python3
"""Convert Jupyter notebooks into marimo examples.

Convert one notebook at a time, or in batch from a list of paths. Creates
diagnostic output in marimo/convert/<name>/.logs/.

Usage:
  convert-colab-to-marimo.py notebook.ipynb --name example-name
  convert-colab-to-marimo.py notebook.ipynb --name example-name --force
  convert-colab-to-marimo.py notebook-path-list.txt --batch
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

MARIMO_DIR = Path("marimo")


class PrepareError(RuntimeError):
    """User-facing setup or input error."""


def validate_repo_root(repo_root: Path) -> None:
    """Validate that ``repo_root`` contains the marimo examples directory.

    Args:
        repo_root: Expected root of the examples repository.

    Raises:
        PrepareError: If ``repo_root`` does not contain ``marimo/``.
    """

    if not (repo_root / MARIMO_DIR).is_dir():
        raise PrepareError(
            "Run this script from the root of the wandb/examples repository "
            "(the directory containing examples/marimo)."
        )


def resolve_file(raw_path: str, *, base_dir: Path) -> Path:
    """Resolve an input file path.

    Args:
        raw_path: Absolute path, user path, or path relative to ``base_dir``.
        base_dir: Directory used to resolve relative paths.

    Returns:
        Absolute resolved path to an existing file.

    Raises:
        PrepareError: If the resolved path is not a file.
    """

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()

    if not path.is_file():
        raise PrepareError(f"input file does not exist: {raw_path}")
    return path


def display_path(path: Path, repo_root: Path) -> str:
    """Format a path for logs and terminal output.

    Args:
        path: Path to display.
        repo_root: Repository root used for relative display.

    Returns:
        Repository-relative path when possible; otherwise an absolute path.
    """

    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def slug_from_notebook(notebook: Path) -> str:
    """Create a batch-mode example name from a notebook filename.

    Args:
        notebook: Source notebook path.

    Returns:
        Lowercase, dash-separated slug derived from the notebook stem.

    Raises:
        PrepareError: If the filename cannot produce a non-empty slug.
    """

    slug = re.sub(r"[^a-z0-9]+", "-", notebook.stem.lower()).strip("-")
    if not slug:
        raise PrepareError(f"could not derive a name from notebook: {notebook}")
    return slug


def validate_name(name: str) -> None:
    """Validate a caller-provided marimo example name.

    Args:
        name: Target directory name under ``marimo/convert``.

    Raises:
        PrepareError: If ``name`` contains unsupported characters.
    """

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise PrepareError(
            "--name must start with a letter or number and contain only "
            f"letters, numbers, '-' or '_': {name}"
        )


def run_command(argv: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a command and capture stdout and stderr together.

    Args:
        argv: Command and arguments to execute.
        cwd: Working directory for the command.

    Returns:
        Completed process, including a 127 return code if the executable is
        missing.
    """

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
    """Write a command transcript for a failed run.

    Args:
        path: Output log file path.
        result: Completed process to serialize.
    """

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
    """Write conversion diagnostics to a JSON file and failure logs.

    Args:
        debug_dir: Directory for ``result.json`` and failure logs.
        source: Source notebook path.
        target: Generated marimo Python file path.
        repo_root: Repository root used for display paths.
        status: Final status string for the conversion attempt.
        failed_stage: Stage name that failed, if any.
        commands: Completed commands keyed by stage name.
    """

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
    """Convert and check one notebook.
    
    Uses ``uvx marimo convert`` and ``uvx marimo check``.

    Args:
        source: Source ``.ipynb`` file.
        name: Target example name under ``marimo/convert``.
        repo_root: Repository root for output paths and command execution.
        force: Whether to overwrite an existing target.

    Returns:
        One of ``"ok"``, ``"conversion_failed"``, or ``"check_failed"``.

    Raises:
        PrepareError: If the input or target name is invalid.
    """

    if source.suffix.lower() != ".ipynb":
        raise PrepareError(f"input must be a .ipynb file: {source}")
    validate_name(name)

    target_dir = repo_root / MARIMO_DIR / "convert" / name
    target = target_dir / f"{name.replace('-', '_')}.py"
    debug_dir = target_dir / ".logs"

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
    """Read notebook paths from a batch file.

    Args:
        path_list: Text file with one notebook path per line.

    Returns:
        Non-empty, non-comment path entries.
    """

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
    """Convert every notebook listed in a batch file.

    Args:
        path_list: Text file containing notebook paths.
        repo_root: Repository root used for resolving batch entries.
        force: Whether to overwrite existing targets.
        fail_on_check: Whether check failures should make the batch fail.

    Returns:
        ``0`` when all required stages pass; otherwise ``1``.
    """

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
    """Build the command-line parser.

    Returns:
        Parser for single-notebook and batch conversion modes.
    """
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
    """Run the CLI.

    Args:
        argv: Optional argument list for tests; defaults to ``sys.argv``.

    Returns:
        Process exit code.
    """

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
