import random
import os

import numpy as np


def set_all_seeds(seed):
    """Seed every RNG the pipeline uses.

    ``PYTHONHASHSEED`` is fixed at interpreter start-up and cannot be changed for the
    running process; setting it here only propagates to child processes (e.g. Spark
    Python workers). No numeric path in this project depends on hash order — all
    container iteration that reaches the features is explicitly sorted — so this is
    defence in depth, not a reproducibility requirement.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return seed
