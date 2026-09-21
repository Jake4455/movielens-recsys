import random
import os

import numpy as np


def set_all_seeds(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return seed
