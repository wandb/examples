"""
benchmark_suite.py -- VAREK Guardrails containment regression battery.

Runs eight payloads spanning four categories (benign, malicious, resource,
edge) and verifies the actual outcome matches the expected containment
behavior. Each payload's full outcome is logged as a row in a W&B Table
for comparison across runs. The script exits non-zero if any payload's
behavior diverges from expectation, making it suitable as a CI gate or
pre-commit hook.

Categories:
  benign     -- payloads that should run to completion (exit_code=0, no containment)
  malicious  -- payloads that should be blocked at the kernel boundary
  resource   -- payloads that should be killed by resource limits
  edge       -- payloads that should fail to parse/execute cleanly

Run with the cgroup wrapper:

    ./scripts/with-cgroup.sh python benchmark_suite.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

import wandb

from varek_guardrails import (
    ExecutionPayload,
    SeccompBpfBackend,
    configure_backend,
    default_python_policy,
    execute_untrusted,
)


PROJECT = os.environ.get("WANDB_PROJECT", "varek-guardrails-benchmarks")
ENTITY = os.environ.get("WANDB_ENTITY")


def is_contained(outcome) -> bool:
    """The kernel boundary fired (or process otherwise didn't exit cleanly)."""
    return (
        outcome.exit_code != 0
        or outcome.killed_by_signal is not None
        or outcome.violation is not None
    )


@dataclass(frozen=True)
class Payload:
    name: str
    category: str
    code: str
    # A predicate over the ExecutionOutcome that returns True when the
    # observed outcome matches expectations. Each payload supplies its own.
    expectation: Callable[[object], bool]
    expectation_description: str


SUITE: list[Payload] = [
    # -- benign ------------------------------------------------------------
    Payload(
        name="benign_arithmetic",
        category="benign",
        code="print(2 + 2)\n",
        expectation=lambda o: o.exit_code == 0 and not is_contained(o),
        expectation_description="exit_code=0, boundary did not fire",
    ),
    Payload(
        name="benign_dict_ops",
        category="benign",
        code=(
            "d = {i: i * 2 for i in range(50)}\n"
            "print(sum(d.values()))\n"
        ),
        expectation=lambda o: o.exit_code == 0 and not is_contained(o),
        expectation_description="exit_code=0, boundary did not fire",
    ),
    # -- malicious ---------------------------------------------------------
    Payload(
        name="malicious_subprocess_run",
        category="malicious",
        code=(
            "import subprocess\n"
            "subprocess.run(['/bin/echo', 'pwned'], check=False)\n"
            "print('should_be_unreachable')\n"
        ),
        expectation=lambda o: is_contained(o) and o.exit_code != 0,
        expectation_description="contained=True, exit_code!=0 (execve denied)",
    ),
    # NOTE: default_python_policy() permits os.system to succeed under the
    # current varek_guardrails 1.1.1 release because /bin/sh is on the default
    # binary_allowlist. This payload is preserved as a documented regression
    # test â if a future policy change blocks /bin/sh execve, the
    # expectation will diverge and force a deliberate update. To block
    # os.system explicitly, construct a stricter policy with a
    # binary_allowlist that excludes /bin/sh.
    Payload(
        name="documented_os_system_passthrough",
        category="known_allowance",
        code=(
            "import os\n"
            "os.system('echo permitted-by-default-policy')\n"
            "print('completed_via_os_system')\n"
        ),
        expectation=lambda o: o.exit_code == 0 and not is_contained(o),
        expectation_description="exit_code=0, boundary did not fire (default policy permits /bin/sh)",
    ),
    Payload(
        name="malicious_socket_connect",
        category="malicious",
        code=(
            "import socket\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "s.connect(('1.1.1.1', 80))\n"
            "print('connected')\n"
        ),
        expectation=lambda o: is_contained(o) and o.exit_code != 0,
        expectation_description="contained=True, exit_code!=0 (network denied)",
    ),
    # -- resource ----------------------------------------------------------
    Payload(
        name="resource_infinite_loop",
        category="resource",
        code="while True:\n    pass\n",
        expectation=lambda o: o.timed_out and is_contained(o),
        expectation_description="timed_out=True, boundary fired",
    ),
    # -- edge --------------------------------------------------------------
    Payload(
        name="edge_syntax_error",
        category="edge",
        code="def main(:\n    return 1\n",  # intentionally malformed
        expectation=lambda o: is_contained(o) and o.exit_code != 0,
        expectation_description="contained=True, exit_code!=0 (parse failure)",
    ),
    Payload(
        name="edge_runtime_exception",
        category="edge",
        code="raise RuntimeError('expected by smoke test')\n",
        expectation=lambda o: is_contained(o) and o.exit_code != 0,
        expectation_description="contained=True, exit_code!=0 (uncaught exception)",
    ),
]


def run_suite() -> tuple[wandb.Table, int]:
    """Execute every payload and return (wandb.Table, divergence_count)."""
    table = wandb.Table(
        columns=[
            "name",
            "category",
            "expected",
            "matched",
            "exit_code",
            "contained",
            "timed_out",
            "killed_by_signal",
            "violation",
            "wall_clock_s",
        ]
    )

    divergences = 0

    for payload in SUITE:
        ep = ExecutionPayload(
            interpreter_path=sys.executable, code=payload.code
        )
        outcome = execute_untrusted(ep, default_python_policy())
        matched = payload.expectation(outcome)
        contained = is_contained(outcome)
        if not matched:
            divergences += 1

        marker = "OK" if matched else "DIVERGED"
        print(
            f"  [{marker}] {payload.name} ({payload.category}): "
            f"exit={outcome.exit_code} "
            f"contained={contained} "
            f"timed_out={outcome.timed_out} "
            f"wall={outcome.wall_clock_s:.2f}s"
        )

        table.add_data(
            payload.name,
            payload.category,
            payload.expectation_description,
            matched,
            outcome.exit_code,
            contained,
            bool(outcome.timed_out),
            outcome.killed_by_signal
            if outcome.killed_by_signal is not None
            else -1,
            outcome.violation if outcome.violation is not None else "",
            outcome.wall_clock_s,
        )

    return table, divergences


def main() -> int:
    configure_backend(SeccompBpfBackend())

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type="benchmark",
        config={
            "n_payloads": len(SUITE),
            "categories": sorted({p.category for p in SUITE}),
            "backend": "SeccompBpfBackend",
        },
    )

    try:
        print(
            f"[benchmark] running {len(SUITE)} payloads against "
            f"default_python_policy()..."
        )
        table, divergences = run_suite()

        match_rate = (len(SUITE) - divergences) / len(SUITE)
        run.log({"benchmark/results": table})
        run.summary["benchmark/match_rate"] = match_rate
        run.summary["benchmark/divergences"] = divergences
        run.summary["benchmark/all_matched"] = divergences == 0

        print(
            f"\n[benchmark] match_rate={match_rate:.2%} "
            f"({len(SUITE) - divergences}/{len(SUITE)})"
        )
        print(f"[benchmark] view at {run.url}")
    finally:
        run.finish()

    return 0 if divergences == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
