# VAREK Guardrails + Weights & Biases

Three runnable examples demonstrating how to integrate
[VAREK Guardrails](https://github.com/kwdoug63/varek) — a Python runtime
containment layer that bounds untrusted code at the Linux kernel via
seccomp-bpf — with W&B for experiment tracking, telemetry, and CI gating.

## What this directory contains

| File                       | Purpose                                                                                              |
| -------------------------- | ---------------------------------------------------------------------------------------------------- |
| `verification_artifact.py` | Runs the public-API verification battery, uploads the structured manifest as a W&B `Artifact`. CI gate. |
| `telemetry_stream.py`      | Bridges VAREK's `subscribe_telemetry` events into `wandb.log`, producing a kernel-event timeline.    |
| `benchmark_suite.py`       | Eight-payload regression battery (benign / malicious / resource / edge), logs a sortable `wandb.Table`. |
| `scripts/with-cgroup.sh`   | Idempotent wrapper that ensures cgroup v2 controllers are enabled before the script runs.            |
| `requirements.txt`         | Pinned floors: `varek-guardrails>=1.1.1`, `wandb>=0.16`, `weave>=0.50`.                              |

## Why this integration

W&B answers *what happened during the run.* VAREK Guardrails answers
*was the run permitted to do what it tried.* For agentic workloads where
an LLM emits code that subsequently executes, the second question is the
one that matters when something goes wrong — and a W&B Artifact attached
to the run is the simplest auditable record that the kernel boundary
fired (or did not).

These examples wire the two together at three different points: as a
verification artifact (CI gate), as live telemetry (timeline), and as a
benchmark battery (regression suite).

## Prerequisites

### Linux kernel + cgroup v2

VAREK Guardrails enforces per-execution memory and CPU limits via cgroup
v2. Required:

- Linux kernel 5.10+
- cgroup v2 mounted at `/sys/fs/cgroup` (default on Ubuntu 22.04+,
  Debian 11+, Fedora 31+, RHEL 9+)
- `libseccomp` available to the Python runtime
- Root or `CAP_SYS_ADMIN` for the user running the examples (cgroup
  controllers in `subtree_control` require it)

### Cgroup controller enablement

VAREK Guardrails creates per-execution sub-cgroups under `varek.slice`.
For those sub-cgroups to support `memory.max` / `cpu.max` / `pids.max`,
the parent slice must have those controllers enabled in
`cgroup.subtree_control`. The bundled `scripts/with-cgroup.sh` wrapper
handles this idempotently — always run the examples through it:

```bash
chmod +x scripts/with-cgroup.sh
./scripts/with-cgroup.sh python verification_artifact.py
./scripts/with-cgroup.sh python telemetry_stream.py
./scripts/with-cgroup.sh python benchmark_suite.py
```

If you skip the wrapper on a system where controllers aren't already
enabled, you'll see:

```
PermissionError: [Errno 13] Permission denied:
'/sys/fs/cgroup/varek.slice/exec-XXXX/memory.max'
```

That is the failure mode the wrapper prevents.

### Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login
```

## Quick start

After installing requirements and authenticating with W&B:

```bash
chmod +x scripts/with-cgroup.sh
./scripts/with-cgroup.sh python verification_artifact.py
```

Expected output: five checks pass, a `varek-guardrails-verification`
artifact is logged to your W&B project, the script exits 0. Open the
run URL printed at the end to inspect the manifest.

To run the full demonstration sequence:

```bash
./scripts/with-cgroup.sh python verification_artifact.py
./scripts/with-cgroup.sh python telemetry_stream.py
./scripts/with-cgroup.sh python benchmark_suite.py
```

## Configuring the W&B project / entity

Each script reads `WANDB_PROJECT` and `WANDB_ENTITY` from the environment.
Defaults:

| Script                    | Default project                  |
| ------------------------- | -------------------------------- |
| `verification_artifact.py` | `varek-guardrails-verification`  |
| `telemetry_stream.py`     | `varek-guardrails-telemetry`     |
| `benchmark_suite.py`      | `varek-guardrails-benchmarks`    |

To override:

```bash
WANDB_PROJECT=my-project WANDB_ENTITY=my-team \
  ./scripts/with-cgroup.sh python verification_artifact.py
```

## Using as a CI gate

Both `verification_artifact.py` and `benchmark_suite.py` exit non-zero on
failure, making them drop-in CI gates:

```yaml
# .github/workflows/varek-verification.yml
- name: VAREK Guardrails verification
  run: ./scripts/with-cgroup.sh python verification_artifact.py
  env:
    WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
```

Note: GitHub-hosted runners do not expose the cgroup v2 delegation
required by `varek_guardrails`. Use a self-hosted Linux runner (or a
container with the appropriate cgroup mounts) for CI.

## Platform support

| Platform              | Status                                                              |
| --------------------- | ------------------------------------------------------------------- |
| Linux (cgroup v2)     | Full enforcement — execve, network, wallclock, memory               |
| macOS                 | `SeccompBpfBackend.is_available()` returns an explanatory string; the package fails closed rather than silently degrading |
| Windows               | Same fail-closed behavior as macOS                                  |
| Docker / Kubernetes   | Requires the container runtime to expose cgroup v2 to the workload  |

Failing closed on unsupported platforms is itself a correctness property
of VAREK Guardrails — the package will not silently run untrusted code
without containment.

## Further reading

- VAREK repository: <https://github.com/kwdoug63/varek>
- Guardrails threat model: <https://github.com/kwdoug63/varek/blob/main/docs/security/threat-model.md>
- Spec paper (language layer): linked from the main VAREK README
- Issue #223 (the v1.0 weakness this layer fixed): <https://github.com/kwdoug63/varek/issues/223>

## License

MIT — same as the parent `wandb/examples` repository and the upstream
VAREK project.
