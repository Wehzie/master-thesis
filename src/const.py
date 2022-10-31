from typing import Final
import numpy as np
GLOBAL_SEED: Final = 5
RNG: Final = np.random.default_rng(GLOBAL_SEED)
MULTIPROCESSING: Final = False
TEST_RNG: Final = np.random.default_rng() # use no seed for testing to increase probability of finding errors
SAMPLE_FLUSH_PERIOD: Final = 1000 # number of samples to store in RAM before flush to disk
LEGAL_DISTS: Final = [RNG.uniform.__name__, RNG.normal.__name__] # supported distributions to draw from