#!/bin/bash
# with-cgroup.sh -- ensure cgroup v2 controllers are enabled in varek.slice
# before delegating to the wrapped command.
#
# This is idempotent and safe to run unconditionally. The 2>/dev/null and
# || true guards make it a no-op when controllers are already enabled or
# when running on a system without cgroup v2 (where varek_guardrails will
# fail closed at the kernel-availability check anyway).
#
# Usage:
#   ./scripts/with-cgroup.sh python verification_artifact.py
#   ./scripts/with-cgroup.sh python benchmark_suite.py

set -e

# Ensure varek.slice exists in the cgroup tree.
mkdir -p /sys/fs/cgroup/varek.slice 2>/dev/null || true

# Enable controllers needed for memory/cpu/pids enforcement on per-execution
# sub-cgroups. If already enabled, the kernel returns EBUSY which we ignore.
echo "+memory +cpu +pids" > /sys/fs/cgroup/varek.slice/cgroup.subtree_control 2>/dev/null || true

# Delegate to the wrapped command with all arguments preserved.
exec "$@"
