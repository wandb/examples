"""
verification_artifact.py -- VAREK Guardrails verification report uploaded
as a W&B Artifact.

Exercises the public VAREK Guardrails API end-to-end and produces a
structured verification manifest attached to a W&B run as an artifact.
This is intended to be run as a CI gate: any FAIL in the manifest exits
non-zero, and the artifact is preserved on the W&B run for audit.

Output artifact: `varek-guardrails-verification` (type: verification-report)

Run with the cgroup wrapper to ensure controllers are enabled:

    ./scripts/with-cgroup.sh python verification_artifact.py
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb

from varek_guardrails import (
    ExecutionPayload,
    IsolationError,
    SeccompBpfBackend,
    configure_backend,
    default_python_policy,
    execute_untrusted,
)


PROJECT = os.environ.get("WANDB_PROJECT", "varek-guardrails-verification")
ENTITY = os.environ.get("WANDB_ENTITY")  # optional; uses default if unset


def is_contained(outcome) -> bool:
    """Derive containment from outcome fields.

    The containment boundary fires when the contained code does not exit
    cleanly. A benign payload that runs to completion is NOT contained --
    there was nothing to contain. This matches the working demo's logic
    in 16-wandb-pipeline-verification-intercept.py.
    """
    return (
        outcome.exit_code != 0
        or outcome.killed_by_signal is not None
        or outcome.violation is not None
    )


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    measurements: dict[str, Any] = field(default_factory=dict)


def _record_outcome_fields(outcome) -> dict[str, Any]:
    """Capture every field on ExecutionOutcome plus derived containment."""
    return {
        "exit_code": outcome.exit_code,
        "killed_by_signal": outcome.killed_by_signal,
        "violation": outcome.violation,
        "timed_out": outcome.timed_out,
        "wall_clock_s": outcome.wall_clock_s,
        "contained_derived": is_contained(outcome),
        "stdout_len": len(outcome.stdout) if outcome.stdout else 0,
        "stderr_len": len(outcome.stderr) if outcome.stderr else 0,
    }


def _build_payload(code: str) -> ExecutionPayload:
    """Construct a payload using the active interpreter."""
    return ExecutionPayload(interpreter_path=sys.executable, code=code)


def check_backend_available() -> CheckResult:
    """SeccompBpfBackend reports itself ready on a conforming host."""
    backend = SeccompBpfBackend()
    reason = backend.is_available()
    if reason is None:
        return CheckResult("backend_available", True, "SeccompBpfBackend ready")
    return CheckResult(
        "backend_available", False, f"backend unavailable: {reason}"
    )


def check_configure_succeeds() -> CheckResult:
    """configure_backend accepts a ready backend without raising."""
    try:
        configure_backend(SeccompBpfBackend())
        return CheckResult("configure_succeeds", True)
    except IsolationError as e:
        return CheckResult("configure_succeeds", False, f"raised: {e}")


def check_benign_payload() -> CheckResult:
    """A pure-computation script runs to completion without triggering containment."""
    payload = _build_payload(
        "total = sum(range(100))\n"
        "print(f'total={total}')\n"
    )
    outcome = execute_untrusted(payload, default_python_policy())
    fields = _record_outcome_fields(outcome)

    # Benign: clean exit AND containment did NOT fire.
    if outcome.exit_code == 0 and not is_contained(outcome):
        return CheckResult(
            "benign_payload", True, "ran cleanly, boundary did not fire", fields
        )
    return CheckResult(
        "benign_payload",
        False,
        f"unexpected exit_code={outcome.exit_code} contained={is_contained(outcome)}",
        fields,
    )


def check_execve_denied() -> CheckResult:
    """A subprocess.run attempt is intercepted at the kernel boundary."""
    payload = _build_payload(
        "import subprocess\n"
        "subprocess.run(['/bin/echo', 'should not run'], check=True)\n"
        "print('reached forbidden code path')\n"
    )
    outcome = execute_untrusted(payload, default_python_policy())
    fields = _record_outcome_fields(outcome)

    # The seccomp-bpf filter denies execve; subprocess raises in
    # _execute_child, the script exits non-zero, containment fires.
    if is_contained(outcome) and outcome.exit_code != 0:
        return CheckResult(
            "execve_denied",
            True,
            "subprocess attempt contained at kernel boundary",
            fields,
        )
    return CheckResult(
        "execve_denied",
        False,
        f"containment did not fire: exit_code={outcome.exit_code}",
        fields,
    )


def check_wallclock_enforced() -> CheckResult:
    """An infinite loop is killed at the configured wall-clock limit."""
    payload = _build_payload("while True:\n    pass\n")
    outcome = execute_untrusted(payload, default_python_policy())
    fields = _record_outcome_fields(outcome)

    if outcome.timed_out and is_contained(outcome):
        return CheckResult(
            "wallclock_enforced",
            True,
            f"killed at {outcome.wall_clock_s:.2f}s",
            fields,
        )
    return CheckResult(
        "wallclock_enforced",
        False,
        f"not timed out: timed_out={outcome.timed_out}",
        fields,
    )


CHECKS = [
    check_backend_available,
    check_configure_succeeds,
    check_benign_payload,
    check_execve_denied,
    check_wallclock_enforced,
]


def build_manifest(results: list[CheckResult]) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
        },
        "checks": [asdict(r) for r in results],
    }


def main() -> int:
    print("[verify] Running VAREK Guardrails verification battery...")
    results: list[CheckResult] = []
    for check_fn in CHECKS:
        try:
            r = check_fn()
        except Exception as exc:  # noqa: BLE001 -- record uncaught failures
            r = CheckResult(check_fn.__name__, False, f"uncaught: {exc!r}")
        marker = "PASS" if r.passed else "FAIL"
        print(f"  [{marker}] {r.name}: {r.detail}")
        results.append(r)

    manifest = build_manifest(results)
    summary = manifest["summary"]

    print(
        f"\n[verify] {summary['passed']}/{summary['total']} passed "
        f"({summary['failed']} failed)"
    )

    # Persist the manifest to a tempfile so wandb.Artifact can include it.
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "verification_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        run = wandb.init(
            project=PROJECT,
            entity=ENTITY,
            job_type="verification",
            config={"manifest_schema": manifest["schema_version"]},
        )
        try:
            artifact = wandb.Artifact(
                name="varek-guardrails-verification",
                type="verification-report",
                description=(
                    "Structured manifest of VAREK Guardrails public-API "
                    "verification checks executed against a live host."
                ),
                metadata=summary,
            )
            artifact.add_file(str(manifest_path), name="manifest.json")
            run.log_artifact(artifact)
            run.summary["verification/passed"] = summary["passed"]
            run.summary["verification/failed"] = summary["failed"]
            run.summary["verification/all_green"] = summary["failed"] == 0
            print(f"[verify] Artifact logged to run {run.url}")
        finally:
            run.finish()

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
