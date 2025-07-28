import os
import sys


def _silence_worker(worker_id):
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')