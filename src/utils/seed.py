import random
import numpy as np


def set_seed(seed):
    if seed is None:
        seed = random.SystemRandom().randrange(2**32)

    random.seed(seed)
    np.random.seed(seed)

    print(f"Seed: {seed}")
    return seed
