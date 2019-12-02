#!/usr/bin/env python
from __future__ import print_function
import subprocess
import signal
import functools
import sys
import os

LOGNAME="wrap_mpi_inner"

# remove program from commandline so we can pass everything else through
_ = sys.argv.pop(0)
if len(sys.argv) == 0:
    print("%s - ERROR: argument parameters required" % LOGNAME, file=sys.stderr)
    sys.exit(2)

def handle_signal(proc, sig, _):
    print("%s - INFO: SIGUSR1 received" % LOGNAME, file=sys.stderr)
    #proc.terminate()
    os.kill(proc.pid, signal.SIGINT)

# os.setsid starts a new session so SIGUSR1 doesnt kill the child
p = subprocess.Popen(sys.argv, stdin=0, stdout=1, stderr=2,
                     preexec_fn=os.setsid)

signal.signal(signal.SIGUSR1, functools.partial(handle_signal, p))
ret = p.wait()

print("%s - INFO: Child exit (%d)" % (LOGNAME, ret), file=sys.stderr)
if ret < 0:
    sys.exit(1)
