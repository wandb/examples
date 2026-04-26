"""
telemetry_stream.py — bridge VAREK Guardrails telemetry into W&B run logs.

VAREK Guardrails emits PEP 578-style audit events through `subscribe_telemetry`.
This example subscribes a callback that buffers every event and flushes them
into the active W&B run as structured log entries between payloads.

Three demonstration payloads:

  1. A benign payload that completes cleanly (boundary does not fire).
  2. A payload that attempts subprocess execution (execve denied).
  3. A payload that runs an infinite loop (wallclock enforced).

The result is a W&B run timeline showing the kernel boundary firing and
what each payload tried to do.

POST-FORK SAFETY NOTE:
  Audit hooks installed via subscribe_telemetry are inherited by forked
  children. VAREK Guardrails uses subprocess.Popen with a preexec_fn
  (seccomp + cgroup setup) that makes ctypes calls — each ctypes call
  triggers our audit hook in the child. Calling wandb.log directly from
  that context breaks the preexec_fn because wandb's threading state does
  not survive fork. We guard the callback with an os.getpid() check so it
  is a no-op in any forked child, and we buffer events to a thread-safe
  queue that is drained from the main thread between payloads.

Run with the cgroup wrapper:

    ./scripts/with-cgroup.sh python telemetry_stream.py
"""

from __future__ import annotations

import os
import queue
import sys
from dataclasses import dataclass
from typing import Any

import wandb

from varek_guardrails import (
    ExecutionPayload,
    SeccompBpfBackend,
    configure_backend,
    default_python_policy,
    execute_untrusted,
    subscribe_telemetry,
)


PROJECT = os.environ.get("WANDB_PROJECT", "varek-guardrails-telemetry")
ENTITY = os.environ.get("WANDB_ENTITY")

# Captured at import time. Any audit-hook callback running with a different
# pid is in a forked child and must be a no-op.
_PARENT_PID = os.getpid()

# Thread- and fork-safe event buffer drained between payloads.
_event_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()


def is_contained(outcome) -> bool:
    """The kernel boundary fired (or process otherwise didn't exit cleanly)."""
    return (
        outcome.exit_code != 0
        or outcome.killed_by_signal is not None
        or outcome.violation is not None
    )


@dataclass
class DemoPayload:
    name: str
    code: str
    expected_summary: str


PAYLOADS: list[DemoPayload] = [
    DemoPayload(
        name="benign_compute",
        code=(
            "total = sum(range(1000))\n"
            "print({'total': total})\n"
        ),
        expected_summary="exit_code=0, no boundary events",
    ),
    DemoPayload(
        name="execve_attempt",
        code=(
            "import subprocess\n"
            "subprocess.run(['/bin/echo', 'attempt'], check=False)\n"
            "print('reached')\n"
        ),
        expected_summary="execve denied at kernel; subprocess.Popen audit event",
    ),
    DemoPayload(
        name="wallclock_overrun",
        code="while True:\n    pass\n",
        expected_summary="killed at 30s wallclock limit",
    ),
]


def telemetry_callback(event: str, args: tuple[Any, ...]) -> None:
    """Audit-hook callback. MUST be safe in post-fork-pre-exec contexts.

    Guards:
      - Skip if pid != parent (we are in a forked child; doing anything
        non-trivial here would break the subprocess preexec_fn).
      - Trap and swallow every exception so a logging bug never propagates
        out of an audit-hook context (which would corrupt sandbox setup).
      - Use queue.put_nowait, which is implemented in C and atomic.
    """
    try:
        if os.getpid() != _PARENT_PID:
            return
        _event_queue.put_nowait((event, repr(args)[:512]))
    except Exception:
        # Never raise from an audit hook.
        pass


def drain_events_to_run(run, label: str) -> int:
    """Flush all buffered events into the active W&B run. Returns count."""
    drained: list[tuple[str, str]] = []
    try:
        while True:
            drained.append(_event_queue.get_nowait())
    except queue.Empty:
        pass

    for event, args_repr in drained:
        run.log(
            {
                "telemetry/event": event,
                "telemetry/args_repr": args_repr,
                "telemetry/payload_label": label,
            }
        )

    return len(drained)


def run_payload(name: str, code: str, run) -> None:
    print(f"\n[telemetry] running payload: {name}")
    payload = ExecutionPayload(interpreter_path=sys.executable, code=code)
    outcome = execute_untrusted(payload, default_python_policy())

    contained = is_contained(outcome)
    n_events = drain_events_to_run(run, label=name)

    # Per-payload outcome fields, namespaced so multiple runs are comparable.
    run.log(
        {
            f"payload/{name}/exit_code": outcome.exit_code,
            f"payload/{name}/contained": int(contained),
            f"payload/{name}/timed_out": int(outcome.timed_out),
            f"payload/{name}/wall_clock_s": outcome.wall_clock_s,
            f"payload/{name}/killed_by_signal": (
                outcome.killed_by_signal
                if outcome.killed_by_signal is not None
                else -1
            ),
            f"payload/{name}/n_telemetry_events": n_events,
        }
    )

    print(
        f"  exit_code={outcome.exit_code} "
        f"contained={contained} "
        f"timed_out={outcome.timed_out} "
        f"wall_clock_s={outcome.wall_clock_s:.3f} "
        f"telemetry_events={n_events}"
    )


def main() -> int:
    configure_backend(SeccompBpfBackend())
    subscribe_telemetry(telemetry_callback)

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type="telemetry-demo",
        config={"backend": "SeccompBpfBackend", "n_payloads": len(PAYLOADS)},
    )

    try:
        print(f"[telemetry] streaming events to {run.url}")

        for p in PAYLOADS:
            run_payload(p.name, p.code, run)

        # Final drain in case any events landed after the last payload.
        residual = drain_events_to_run(run, label="post_run")
        if residual:
            print(f"[telemetry] flushed {residual} residual events post-run")

        run.summary["telemetry/payloads_run"] = len(PAYLOADS)
        print(f"\n[telemetry] complete. View timeline at {run.url}")
    finally:
        run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
