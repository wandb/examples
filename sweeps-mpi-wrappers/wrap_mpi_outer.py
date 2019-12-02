#!/usr/bin/env python
from __future__ import print_function
import subprocess
import signal
import functools
import sys
import os

LOGNAME="wrap_mpi_outer"

split_command = [
    # add your commandline here
    'mpirun', '-np', '1',
    'python', 'wrap_mpi_inner.py',
    'python', 'train.py',
]

def handle_signal(proc, sig, _):
    print("%s - INFO: SIGTERM received" % LOGNAME, file=sys.stderr)
    #proc.terminate()
    os.kill(proc.pid, signal.SIGUSR1)

# os.setsid starts a new session so SIGUSR1 doesnt kill the child
p = subprocess.Popen(split_command, stdin=0, stdout=1, stderr=2)

signal.signal(signal.SIGTERM, functools.partial(handle_signal, p))
ret = p.wait()

print("%s - INFO: Child exit (%d)" % (LOGNAME, ret), file=sys.stderr)
if ret < 0:
    sys.exit(1)
