import os
import sys

import numpy as np
import torch


def _silence_worker(worker_id):
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    # Seed numpy RNG per worker so each produces unique random samples
    np.random.seed(torch.initial_seed() % (2**32))