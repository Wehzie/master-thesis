import numpy as np
GLOBAL_SEED = 5
RNG = np.random.default_rng(GLOBAL_SEED)
TEST_RNG = np.random.default_rng() # use no seed for testing to increase probability of finding errors
SAMPLE_FLUSH_PERIOD = 1000 # number of samples to store in RAM before flush to disk
LEGAL_DISTS = [RNG.uniform.__name__, RNG.normal.__name__] # supported distributions to draw from